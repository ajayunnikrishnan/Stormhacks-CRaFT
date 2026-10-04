"""fp64 CPU reference for the curved walker (web/shaders/walk_curved.frag), docs/WRITEUP.md
§3.4–3.6, §3.11. Same control flow as the shader; used to separate GPU fp32 precision
effects from genuine geometry when validating curved renders.

Usage:
    python tools/ref_render_curved.py web/public/scenes/synth_room --k -0.05 --camera 0 --width 96 --height 72 --out ref.png --npy ref.npy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import curved as cv
from pf_common import edges_to_csr
from ref_render import ExportedScene, sv_colours  # noqa: F401  (flat SV kept for reference)

TEMP = 10.0
TWO_PI = 2 * np.pi


# ---------------------------------------------------------------- geometry (fp64 mirrors of space.ts)
def geodesic(k, o, v, t):
    return cv.cs(k, t) * o + cv.sn(k, t) * v


def geodesic_dir(k, o, v, t):
    return -k * cv.sn(k, t) * o + cv.cs(k, t) * v


def tangent_toward(k, x, y):
    d = float(cv.distance(k, x, y))
    s = cv.sn(k, d)
    if s < 1e-300:
        return np.zeros(4), d
    return (y - cv.cs(k, d) * x) / s, d


def plane_exit_after(k, A, B, t_min):
    return cv.plane_exit_after(k, A, B, t_min)


def plane_entry_before(k, A, B, t_max):
    if k > 0:
        R = np.hypot(A, B)
        if R < 1e-12:
            return -np.inf
        base = np.arctan2(B, A) + np.pi / 2
        return base + TWO_PI * np.floor((t_max - base) / TWO_PI)
    if k == 0:
        if B >= 0:
            return -np.inf
        t = -A / B
        return t if t <= t_max else -np.inf
    if abs(B) <= abs(A) or B >= 0:
        return -np.inf
    t = 0.5 * np.log((B - A) / (A + B))
    return t if t <= t_max else -np.inf


def ball_interval(k, o, v, p, r, t_min):
    if k == 0:
        oc = o[1:] - p[1:]
        qb = 2 * np.dot(oc, v[1:])
        qc = np.dot(oc, oc) - r * r
        disc = qb * qb - 4 * qc
        if disc < 0:
            return False, 0.0, 0.0
        s = np.sqrt(disc)
        return True, (-qb - s) / 2, (-qb + s) / 2
    A = cv.form(k, o, p)
    B = cv.form(k, v, p)
    if k < 0:
        cr = np.cosh(r)
        disc = cr * cr - (A * A - B * B)
        if disc < 0:
            return False, 0.0, 0.0
        s = np.sqrt(disc)
        E1, E2 = (-cr + s) / (A + B), (-cr - s) / (A + B)
        return True, np.log(E1), np.log(E2)
    R = np.hypot(A, B)
    cr = np.cos(r)
    if R < cr:
        return False, 0.0, 0.0
    alpha = np.arccos(min(1.0, cr / R))
    phi = np.arctan2(B, A)
    kk = np.ceil((t_min - (phi + alpha)) / TWO_PI)
    return True, phi - alpha + TWO_PI * kk, phi + alpha + TWO_PI * kk


def log_at_origin(k, p):
    if k == 0:
        return p[1:].copy()
    d = float(cv.distance(k, np.array([1.0, 0, 0, 0]), p))
    s = cv.sn(k, d)
    f = 1.0 if s < 1e-300 else d / s
    return f * p[1:]


def inverse_isometry(k, M):
    if k == 0:
        R = M[1:, 1:]
        out = np.eye(4)
        out[1:, 1:] = R.T
        out[1:, 0] = -R.T @ M[1:, 0]
        return out
    J = np.diag([k, 1, 1, 1.0])
    return J @ M.T @ J


# ---------------------------------------------------------------- curved scene state
class CurvedRef:
    def __init__(self, ex: ExportedScene, kv: float):
        self.ex = ex
        self.k = int(np.sign(kv)) if abs(kv) >= 1e-9 else 0
        self.s = float(np.sqrt(abs(kv))) if self.k else 1.0
        man = ex.manifest
        self.centre = np.array(man["curved"]["centre"]) if "curved" in man else 0.5 * (np.array(man["bbox_min"]) + np.array(man["bbox_max"]))
        self.sites = cv.CurvedSites(self.k, self.s, ex.pos.astype(np.float64), ex.radius.astype(np.float64), self.centre)
        sec = man["sections"]
        if "adjoff_u" in sec:
            blob = (Path(ex_dir_of(ex)) / man["bin"]).read_bytes()
            self.off = np.frombuffer(blob, np.uint32, count=sec["adjoff_u"]["shape"][0], offset=sec["adjoff_u"]["offset"]).astype(np.int64)
            self.idx = np.frombuffer(blob, np.uint32, count=sec["adjidx_u"]["shape"][0], offset=sec["adjidx_u"]["offset"]).astype(np.int64)
        else:
            self.off, self.idx = ex.adj_off, ex.adj_idx
        self.kk = 1 if self.k == 0 else self.k
        # per-cell frames
        self.P = self.sites.P
        self.R = self.sites.R
        self.M = [cv.translation_to(self.k, self.P[i]) for i in range(ex.n)]
        self.Minv = [inverse_isometry(self.k, M) for M in self.M]

    def site_rgb(self, o):
        """sv_prepass_curved: per detail site colour for camera at world point o."""
        ex, k, s = self.ex, self.k, self.s
        out = np.empty((ex.n, ex.k, 3), np.float64)
        temp = np.linalg.norm(ex.sv_axis, axis=-1).astype(np.float64)
        axis = ex.sv_axis.astype(np.float64) / np.maximum(temp[..., None], 1e-20)
        rgbv = ex.sv_rgb.astype(np.float64)
        for i in range(ex.n):
            M, Minv = self.M[i], self.Minv[i]
            for j in range(ex.k):
                S = M @ cv.embed(k, s * ex.site_off[i, j].astype(np.float64))
                u, d = tangent_toward(k, o, S)
                arrive = geodesic_dir(k, o, u, d)
                loc = Minv @ arrive
                dirn = loc[1:] / np.linalg.norm(loc[1:])
                w = np.exp(-temp[i, j] * np.linalg.norm(dirn[None, :] - axis[i, j], axis=-1))
                out[i, j] = np.maximum(np.sum(w[:, None] * rgbv[i, j], 0) / w.sum() + 0.5, 0)
        return out

    def start_cell(self, o):
        return int(self.sites.cell_of(o[None])[0])

    def trace(self, o, v, start, site_rgb, thr=1e-2, near_cull=False, t_max=None, max_steps=512):
        ex, k, s = self.ex, self.k, self.s
        if t_max is None:
            t_max = TWO_PI if k > 0 else np.inf
        rgb = np.zeros(3)
        log_t = 0.0
        prim, pt_near = start, 0.0
        steps = 0
        while steps < max_steps:
            trans = np.exp(log_t)
            if trans < thr or prim < 0 or pt_near > t_max:
                break
            steps += 1
            ai, P, R = self.sites.A[prim], self.P[prim], self.R[prim]
            hit, tb1, tb2 = ball_interval(k, o, v, P, R, pt_near)
            if hit and tb2 < 0:
                hit = False
            t_near, t_far = max(tb1, 0.0), tb2
            if near_cull and cv.distance(k, o, P) < 4 * R:
                hit = False
            nb = self.idx[self.off[prim] : self.off[prim + 1]]
            da = self.sites.A[nb] - ai
            w = da.copy()
            w[:, 0] *= self.kk
            Aarr, Barr = w @ o, w @ v
            nxt, pt_far = -1, np.inf
            for q in range(nb.size):
                te = plane_exit_after(k, Aarr[q], Barr[q], pt_near)
                if te < pt_far:
                    pt_far, nxt = te, int(nb[q])
                t_far = min(te, t_far)
                tn = plane_entry_before(k, Aarr[q], Barr[q], pt_near)
                t_near = max(tn, t_near)
            nflat, sigma_m = ex.normal[prim].astype(np.float64), float(ex.sigma[prim])
            sigma = sigma_m / s
            if (not hit) or t_near > t_far or sigma_m < 1e-3:
                prim, pt_near = nxt, max(pt_near, pt_far)
                continue
            Minv = self.Minv[prim]
            xin = Minv @ geodesic(k, o, v, t_near)
            din = Minv @ geodesic_dir(k, o, v, t_near)
            e3 = log_at_origin(k, xin)
            d3 = din[1:] / np.linalg.norm(din[1:])
            tau_far = t_far - t_near
            dp0 = np.dot(nflat, d3)
            tau_surf0 = -np.dot(e3, nflat) / dp0
            tq0 = 0.0 if dp0 >= 0 else max(0.0, tau_surf0)
            xq0 = e3 + tq0 * d3
            inv_r2 = 1.0 / (R * R)
            sites3 = s * ex.site_off[prim].astype(np.float64)
            heights = s * ex.height[prim].astype(np.float64)
            wgt = np.exp(-TEMP * np.sum((xq0[None] - sites3) ** 2, -1) * inv_r2)
            height = np.sum(wgt * heights) / max(wgt.sum(), 1e-20)
            tau_surf = (height - np.dot(e3, nflat)) / dp0
            tq1 = 0.0 if dp0 >= 0 else max(0.0, tau_surf)
            xq1 = e3 + tq1 * d3
            wgt = np.exp(-TEMP * np.sum((xq1[None] - sites3) ** 2, -1) * inv_r2)
            colour = (wgt[:, None] * site_rgb[prim]).sum(0) / max(wgt.sum(), 1e-20)
            tau_near = 0.0
            if dp0 >= 0:
                tau_far = min(tau_surf, tau_far)
            else:
                tau_near = max(tau_surf, tau_near)
            prim, pt_near = nxt, max(pt_near, pt_far)
            dt = tau_far - tau_near
            if hit and dt > 0:
                delta = -sigma * dt
                alpha = 1 - np.exp(delta)
                rgb += colour * alpha * trans
                log_t += delta
        return rgb, steps


def ex_dir_of(ex: ExportedScene) -> str:
    return ex._dir  # set in load below


def load(scene_dir: str) -> ExportedScene:
    ex = ExportedScene(scene_dir)
    ex._dir = str(scene_dir)
    return ex


def camera_rays_iso(cam: dict, ref: CurvedRef, width: int, height: int, repo_grid=True):
    """IsoCamera.setFromRepoCamera + walk_curved pixel mapping (repo grid: edges at ±1)."""
    k, s, c = ref.k, ref.s, ref.centre
    r = np.array(cam["right"], float)
    u = np.array(cam["up"], float)
    tx, ty = np.linalg.norm(r), np.linalg.norm(u)
    rh, uh = r / tx, u / ty
    f = np.cross(uh, rh)
    f /= np.linalg.norm(f)
    eye_m = np.array(cam["eye"], float)
    pos = cv.embed(k, (eye_m - c) * s)
    T = cv.translation_to(k, pos)
    # camera space: right=+x, up=+y, forward=−z; W = R·T⁻¹; W⁻¹ = T·Rᵀ
    R3 = np.stack([rh, uh, -f], 0)  # rows
    Rt4 = np.eye(4)
    Rt4[1:, 1:] = R3.T
    invW = T @ Rt4
    o = invW @ np.array([1.0, 0, 0, 0])
    i, j = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    if repo_grid:
        px = 2 * j / (width - 1) - 1
        py = 2 * (height - 1 - i) / (height - 1) - 1
    else:
        px = 2 * (j + 0.5) / width - 1
        py = 2 * (height - 1 - i + 0.5) / height - 1
    dcam = np.stack([px * tx, py * ty, -np.ones_like(px)], -1)
    dcam /= np.linalg.norm(dcam, axis=-1, keepdims=True)
    v = np.einsum("ab,ijb->ija", invW[:, 1:], dcam)  # invW · (0, dcam)
    return o, v


def render(ref: CurvedRef, cam: dict, width: int, height: int, thr=1e-2, near_cull=False, progress=False):
    o, dirs = camera_rays_iso(cam, ref, width, height)
    site_rgb = ref.site_rgb(o)
    start = ref.start_cell(o)
    img = np.zeros((height, width, 3))
    steps = np.zeros((height, width), np.int32)
    for i in range(height):
        for j in range(width):
            img[i, j], steps[i, j] = ref.trace(o, dirs[i, j], start, site_rgb, thr, near_cull)
        if progress and i % 8 == 0:
            print(f"row {i}/{height}", flush=True)
    return img.astype(np.float32), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--k", type=float, required=True, help="curvature in 1/m² (sign = κ)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--height", type=int, default=72)
    ap.add_argument("--out", default="ref_curved.png")
    ap.add_argument("--npy", default=None)
    args = ap.parse_args()
    ex = load(args.scene_dir)
    ref = CurvedRef(ex, args.k)
    img, steps = render(ref, ex.cameras[args.camera], args.width, args.height, progress=True)
    print(f"k={args.k} κ={ref.k} s={ref.s:.4f} mean steps {steps.mean():.1f} max {steps.max()}")
    from PIL import Image

    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(args.out)
    if args.npy:
        np.save(args.npy, img)


if __name__ == "__main__":
    main()
