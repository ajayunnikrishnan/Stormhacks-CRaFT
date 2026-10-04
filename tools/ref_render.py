"""Reference CPU ray tracer: a line-by-line numpy port of
`powerfoam/raytrace.py:RayTracer.benchmark_kernel` (+ `rendering_math.py`,
`color_fn.py`). Slow (pure Python per ray) but exact; used to validate the GLSL
port locally where the CUDA reference cannot run. Flat space only.

The order of operations is deliberately identical to the Warp kernel, including
its quirks (see docs/ARCHITECTURE.md §5): near-camera cull |p-eye| < 4r, no
`t_face > pt_near` check, sigma < 1e-3 short-circuit, fp32 arithmetic.

Usage:
    python tools/ref_render.py web/public/scenes/synth_room --camera 0 --width 160 --height 120 --out ref.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

INT_MAX = 0x7FFFFFFF
TEMP = np.float32(10.0)  # raytrace.py:39  temp = wp.constant(10.0)


# --------------------------------------------------------------------------- #
# Scene loading from the exported binary (so we test exactly what the GPU sees)
# --------------------------------------------------------------------------- #
class ExportedScene:
    def __init__(self, scene_dir: str | Path):
        scene_dir = Path(scene_dir)
        self.manifest = json.loads((scene_dir / "scene.json").read_text())
        blob = (scene_dir / self.manifest["bin"]).read_bytes()
        sec = self.manifest["sections"]

        def get(name):
            s = sec[name]
            a = np.frombuffer(blob, dtype=np.dtype(s["dtype"]), count=int(np.prod(s["shape"])), offset=s["offset"])
            return a.reshape(s["shape"])

        self.n, self.k, self.d = self.manifest["n"], self.manifest["k"], self.manifest["d"]
        pos = get("pos").astype(np.float32)
        self.pos, self.radius = pos[:, :3], pos[:, 3]
        ns = get("nsigma").astype(np.float32)
        self.normal, self.sigma = ns[:, :3], ns[:, 3]
        so = get("siteoff").astype(np.float32).reshape(self.n, self.k, 4)
        self.site_off, self.height = so[..., :3], so[..., 3]
        self.sv_axis = get("svaxis").astype(np.float32).reshape(self.n, self.k, self.d, 3)
        self.sv_rgb = get("svrgb").astype(np.float32).reshape(self.n, self.k, self.d, 3)
        self.adj_off = get("adjoff").astype(np.int64)
        self.adj_idx = get("adjidx").astype(np.int64)
        self.cameras = self.manifest.get("cameras", [])

    @property
    def site_pos(self):
        return self.pos[:, None, :] + self.site_off  # (N,k,3)


# --------------------------------------------------------------------------- #
# rendering_math.py ports (float32)
# --------------------------------------------------------------------------- #
def ray_sphere_intersect(eye, d, c, r):
    """rendering_math.py:5 — returns (hit, t_near, t_far)."""
    oc = eye - c
    qb = np.float32(2.0) * np.dot(oc, d)
    qc = np.dot(oc, oc) - r * r
    disc = qb * qb - np.float32(4.0) * qc
    if disc < 0:
        return False, np.float32(0), np.float32(0)
    sq = np.sqrt(disc)
    t_far = (-qb + sq) / np.float32(2.0)
    t_near = (-qb - sq) / np.float32(2.0)
    if t_near < 0 and t_far < 0:
        return False, np.float32(0), np.float32(0)
    if t_near < 0:
        return True, np.float32(0), t_far
    return True, t_near, t_far


def ray_plane_intersect(eye, d, p, n, h=np.float32(0)):
    """rendering_math.py:129 — plane n·x = n·p + h; returns (t, dp)."""
    dp = np.dot(n, d)
    t = (np.dot(p - eye, n) + h) / dp
    return t, dp


def sv_colours(scene: ExportedScene, eye: np.ndarray, forward=None, fov_cos_cutoff=None) -> np.ndarray:
    """color_fn.py:55 spherical_voronoi_fwd_kernel over all N*k detail sites.

    rgb = max( Σ_d w_d rgb_d / Σ_d w_d + 0.5 , 0 ),  w_d = exp(-temp_d |dir - axis_d|),
    dir = normalize(site - eye). Optional frustum cull (repo greys culled sites to 0.5).
    """
    site = scene.site_pos.astype(np.float32)  # (N,k,3)
    dirn = site - eye[None, None, :].astype(np.float32)
    dirn /= np.linalg.norm(dirn, axis=-1, keepdims=True)
    temp = np.linalg.norm(scene.sv_axis, axis=-1)  # (N,k,D)
    axis = scene.sv_axis / np.maximum(temp[..., None], np.float32(1e-20))
    dist = np.linalg.norm(dirn[:, :, None, :] - axis, axis=-1)  # (N,k,D)
    w = np.exp(-temp * dist)
    rgb = np.sum(w[..., None] * scene.sv_rgb, axis=2) / np.sum(w, axis=2)[..., None] + np.float32(0.5)
    rgb = np.maximum(rgb, 0)
    if fov_cos_cutoff is not None:
        culled = np.sum(dirn * forward[None, None, :], -1) < fov_cos_cutoff
        rgb[culled] = 0.5
    return rgb.astype(np.float32)


def plane_intersection_fwd_local(eye, d, t_near, p, n, r, sites, rgbs, heights):
    """raytrace.py:43 — soft-Voronoi displacement + colour at the dipole hit.
    Returns (height_out, t_surf, dp, rgb_out)."""
    _t_surf, _dp = ray_plane_intersect(eye, d, p, n)
    _t_query = t_near if _dp >= 0 else max(t_near, _t_surf)
    xq = eye + _t_query * d
    inv_r2 = np.float32(1.0) / (r * r)
    w = np.exp(-TEMP * np.sum((xq[None, :] - sites) ** 2, -1) * inv_r2)
    wsum = max(np.sum(w), np.float32(1e-20))
    height_out = np.float32(np.sum(w * heights) / wsum)

    t_surf, dp = ray_plane_intersect(eye, d, p, n, height_out)
    t_query = t_near if dp >= 0 else max(t_near, t_surf)
    x = eye + t_query * d
    w = np.exp(-TEMP * np.sum((x[None, :] - sites) ** 2, -1) * inv_r2)
    wsum = max(np.sum(w), np.float32(1e-20))
    rgb_out = (w[:, None] * rgbs).sum(0) / wsum
    return height_out, t_surf, dp, rgb_out


def start_cell(scene: ExportedScene, eye: np.ndarray) -> int:
    d2 = np.sum((scene.pos - eye[None, :]) ** 2, -1)
    return int(np.argmin(d2 - scene.radius**2))


def trace_ray(scene: ExportedScene, site_rgb, eye, d, start, thr=1e-2, near_cull=True, max_steps=100000):
    """raytrace.py:157-239, one ray. Returns (rgb, n_steps)."""
    f32 = np.float32
    eye, d = eye.astype(f32), d.astype(f32)
    rgb = np.zeros(3, f32)
    log_t = f32(0)
    prim = start
    pt_near = f32(0)
    steps = 0
    while True:
        trans = np.exp(log_t)
        if trans < thr or prim == INT_MAX or steps >= max_steps:
            break
        steps += 1
        center, radius = scene.pos[prim], scene.radius[prim]
        hit, t_near, t_far = ray_sphere_intersect(eye, d, center, radius)
        if near_cull and np.linalg.norm(center - eye) < f32(4.0) * radius:
            hit = False
        a, b = scene.adj_off[prim], scene.adj_off[prim + 1]
        nb = scene.adj_idx[a:b]
        # ray_pface_intersect_diff with diff = p_j - p_i, pm_diff = pm_j - pm_i (computed in fp32)
        diff = scene.pos[nb] - center  # (m,3)
        pm_i = f32(0.5) * (np.dot(center, center) - radius * radius)
        pm_j = f32(0.5) * (np.sum(scene.pos[nb] ** 2, -1) - scene.radius[nb] ** 2)
        pm_diff = pm_j - pm_i
        dp = diff @ d
        with np.errstate(divide="ignore", invalid="ignore"):
            t_face = (pm_diff - diff @ eye) / dp
        next_prim = INT_MAX
        pt_far = f32(1e10)
        exits = dp >= 0
        if np.any(exits):
            # first exit face: min t_face among dp>=0 (ties: first index, as in the serial loop)
            cand = np.where(exits, t_face, np.inf)
            j = int(np.argmin(cand))
            if cand[j] < pt_far:
                next_prim = int(nb[j])
                pt_far = f32(cand[j])
            t_far = min(t_far, f32(np.min(cand)))
        if np.any(~exits):
            t_near = max(t_near, f32(np.max(np.where(~exits, t_face, -np.inf))))
        normal, sigma = scene.normal[prim], scene.sigma[prim]
        if (not hit) or t_near > t_far or sigma < f32(1e-3):
            prim = next_prim
            pt_near = max(pt_near, pt_far)
            continue
        _, t_surf, dps, colour = plane_intersection_fwd_local(
            eye, d, t_near, center, normal, radius, scene.site_pos[prim], site_rgb[prim], scene.height[prim]
        )
        if dps >= 0:
            t_far = min(t_surf, t_far)
        else:
            t_near = max(t_surf, t_near)
        prim = next_prim
        pt_near = max(pt_near, pt_far)
        dt = t_far - t_near
        if hit and dt > 0:
            delta = -sigma * dt
            alpha = f32(1.0) - np.exp(delta)
            rgb += colour * alpha * trans
            log_t += delta
    return rgb, steps  # background is black in all shipped configs


def camera_rays(cam: dict, width: int, height: int):
    """camera.py:get_ray_dir: x = 2j/(W-1) - 1, y = 1 - 2i/(H-1), dir = x·right + y·up + forward."""
    right, up = np.array(cam["right"], np.float32), np.array(cam["up"], np.float32)
    fwd = np.cross(up, right)
    fwd /= np.linalg.norm(fwd)
    i, j = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    x = 2 * j / (width - 1) - 1
    y = 1 - 2 * i / (height - 1)
    d = x[..., None] * right + y[..., None] * up + fwd
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d.astype(np.float32), fwd.astype(np.float32)


def render(scene: ExportedScene, cam: dict, width: int, height: int, thr=1e-2, near_cull=True, progress=False):
    eye = np.array(cam["eye"], np.float32)
    dirs, fwd = camera_rays(cam, width, height)
    site_rgb = sv_colours(scene, eye)
    start = start_cell(scene, eye)
    img = np.zeros((height, width, 3), np.float32)
    steps = np.zeros((height, width), np.int32)
    for i in range(height):
        for j in range(width):
            img[i, j], steps[i, j] = trace_ray(scene, site_rgb, eye, dirs[i, j], start, thr, near_cull)
        if progress and i % 8 == 0:
            print(f"row {i}/{height}", flush=True)
    return img, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=160)
    ap.add_argument("--height", type=int, default=120)
    ap.add_argument("--out", default="ref.png")
    ap.add_argument("--npy", default=None)
    ap.add_argument("--no-near-cull", action="store_true")
    args = ap.parse_args()
    scene = ExportedScene(args.scene_dir)
    cam = scene.cameras[args.camera]
    img, steps = render(scene, cam, args.width, args.height, near_cull=not args.no_near_cull, progress=True)
    print(f"mean steps/ray {steps.mean():.1f}, max {steps.max()}")
    from PIL import Image

    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(args.out)
    if args.npy:
        np.save(args.npy, img)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
