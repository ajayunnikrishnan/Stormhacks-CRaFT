"""Generate a synthetic Power Foam scene in the repo's own checkpoint format.

Stand-in for a trained scene until a real `model.pt` arrives: a simple room
(floor, ceiling, four walls, a box, a sphere, a pillar) whose surfaces are
covered with oriented power cells. Every attribute follows the conventions of
`powerfoam/scene.py` so the exporter and renderers cannot tell it apart from a
trained checkpoint (see docs/ARCHITECTURE.md §2).

Usage:
    python tools/synth_scene.py --out scenes/synth_room [--spacing 0.1] [--k 4] [--d 4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pf_common import RawCheckpoint, frame_to_quat, inverse_softplus, save_checkpoint

ROOM_HALF = 4.0
ROOM_H = 3.0


def _tangent_frame(n: np.ndarray, rng: np.random.Generator):
    """Orthonormal (t, b) for each normal n, with a random in-plane rotation."""
    a = np.where(np.abs(n[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    t = np.cross(n, a)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    b = np.cross(n, t)
    phi = rng.uniform(0, 2 * np.pi, n.shape[0])[:, None]
    t2 = np.cos(phi) * t + np.sin(phi) * b
    b2 = np.cross(n, t2)
    return t2, b2


def _grid_quad(origin, u_axis, v_axis, nu, nv, normal, spacing, rng):
    """Sites on a planar quad: origin + (i+0.5)·h·u + (j+0.5)·h·v."""
    i, j = np.meshgrid(np.arange(nu) + 0.5, np.arange(nv) + 0.5, indexing="ij")
    p = origin[None, :] + (i.reshape(-1, 1) * spacing) * u_axis[None, :] + (j.reshape(-1, 1) * spacing) * v_axis[None, :]
    n = np.tile(normal[None, :], (p.shape[0], 1))
    return p, n


def _surfaces(spacing: float, rng: np.random.Generator):
    H, L = ROOM_HALF, ROOM_H
    nx = int(round(2 * H / spacing))
    ny = int(round(L / spacing))
    parts = []  # (points, normals, colour_fn_id)
    ex, ey, ez = np.eye(3)
    # floor (normal +y, dense half below), ceiling (normal -y)
    parts.append((*_grid_quad(np.array([-H, 0.0, -H]), ex, ez, nx, nx, ey, spacing, rng), "floor"))
    parts.append((*_grid_quad(np.array([-H, L, -H]), ex, ez, nx, nx, -ey, spacing, rng), "ceiling"))
    # walls, normals pointing into the room
    parts.append((*_grid_quad(np.array([-H, 0.0, -H]), ex, ey, nx, ny, ez, spacing, rng), "wall_n"))  # z=-H
    parts.append((*_grid_quad(np.array([-H, 0.0, H]), ex, ey, nx, ny, -ez, spacing, rng), "wall_s"))  # z=+H
    parts.append((*_grid_quad(np.array([-H, 0.0, -H]), ez, ey, nx, ny, ex, spacing, rng), "wall_w"))  # x=-H
    parts.append((*_grid_quad(np.array([H, 0.0, -H]), ez, ey, nx, ny, -ex, spacing, rng), "wall_e"))  # x=+H
    # box 1.2 x 1.0 x 1.2 at (2, 0, -2), normals outward
    bc, bs = np.array([2.0, 0.5, -2.0]), np.array([0.6, 0.5, 0.6])
    for axis in range(3):
        for sgn in (-1.0, 1.0):
            nrm = np.zeros(3)
            nrm[axis] = sgn
            u = np.zeros(3)
            u[(axis + 1) % 3] = 1
            v = np.zeros(3)
            v[(axis + 2) % 3] = 1
            nu = int(round(2 * bs[(axis + 1) % 3] / spacing))
            nv = int(round(2 * bs[(axis + 2) % 3] / spacing))
            origin = bc + nrm * bs[axis] - u * bs[(axis + 1) % 3] - v * bs[(axis + 2) % 3]
            parts.append((*_grid_quad(origin, u, v, nu, nv, nrm, spacing, rng), "box"))
    # sphere r=0.7 at (-2, 0.7, 2): Fibonacci points
    sc, sr = np.array([-2.0, 0.7, 2.0]), 0.7
    m = int(round(4 * np.pi * sr * sr / spacing**2))
    idx = np.arange(m) + 0.5
    phi = np.arccos(1 - 2 * idx / m)
    th = np.pi * (1 + 5**0.5) * idx
    nrm = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], -1)
    parts.append((sc + sr * nrm, nrm, "sphere"))
    # pillar (cylinder) r=0.4 at (0, *, -2.5), full height
    pc, pr = np.array([0.0, 0.0, -2.5]), 0.4
    na = int(round(2 * np.pi * pr / spacing))
    ang = (np.arange(na) + 0.5) / na * 2 * np.pi
    ring_n = np.stack([np.cos(ang), np.zeros(na), np.sin(ang)], -1)
    ys = (np.arange(ny) + 0.5) * spacing
    pn = np.tile(ring_n, (ny, 1))
    pp = pc + pr * pn + np.repeat(ys, na)[:, None] * np.array([0, 1.0, 0])
    parts.append((pp, pn, "pillar"))
    return parts


def _base_colour(kind: str, p: np.ndarray) -> np.ndarray:
    """Per-site albedo (0..1) with simple procedural patterns."""
    c = np.zeros((p.shape[0], 3))
    if kind == "floor":
        chk = (np.floor(p[:, 0] / 0.5) + np.floor(p[:, 2] / 0.5)) % 2
        c[:] = np.where(chk[:, None] > 0.5, [0.86, 0.82, 0.74], [0.32, 0.34, 0.40])
    elif kind == "ceiling":
        c[:] = [0.80, 0.80, 0.82]
    elif kind.startswith("wall"):
        hue = {"wall_n": [0.70, 0.45, 0.35], "wall_s": [0.35, 0.55, 0.70], "wall_w": [0.45, 0.65, 0.40], "wall_e": [0.70, 0.65, 0.35]}[kind]
        stripe = (np.floor(p[:, 1] / 0.5) % 2)[:, None]
        c[:] = np.array(hue)[None, :] * (0.75 + 0.25 * stripe)
    elif kind == "box":
        c[:] = [0.90, 0.50, 0.15]
    elif kind == "sphere":
        c[:] = [0.15, 0.65, 0.65]
    elif kind == "pillar":
        c[:] = [0.65, 0.25, 0.20]
    return c


def make_scene(spacing: float, k: int, d: int, seed: int = 0) -> tuple[RawCheckpoint, dict]:
    rng = np.random.default_rng(seed)
    parts = _surfaces(spacing, rng)
    P = np.concatenate([p for p, _, _ in parts], 0)
    Nrm = np.concatenate([n for _, n, _ in parts], 0)
    C = np.concatenate([_base_colour(kind, p) for p, _, kind in parts], 0)
    n = P.shape[0]
    # small jitter avoids exactly coplanar/cospherical sites (qhull degeneracies)
    P = P + rng.normal(0, 0.02 * spacing, P.shape)
    r = 0.9 * spacing * rng.uniform(0.95, 1.05, n)
    sigma = 300.0

    t, b = _tangent_frame(Nrm, rng)
    quat = frame_to_quat(Nrm, t, b)

    # detail sites: in the disc of radius 0.6 r (units of r)
    rho = 0.6 * np.sqrt(rng.uniform(0, 1, (n, k)))
    ang = rng.uniform(0, 2 * np.pi, (n, k))
    texel_sites = np.stack([rho * np.cos(ang), rho * np.sin(ang)], -1)
    shade = rng.uniform(0.85, 1.15, (n, k, 1))
    site_col = np.clip(C[:, None, :] * shade, 0, 1)  # (n,k,3)
    texel_height = rng.uniform(-0.05, 0.05, (n, k))

    # SV: D random axes, temperature 2 (soft), rgb = colour - 0.5 (+ small sheen per axis)
    ax = rng.normal(size=(n, k, d, 3))
    ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
    temp = rng.uniform(1.5, 2.5, (n, k, d, 1))
    sv_axis = (ax * temp).reshape(n, k, 3 * d)
    sheen = rng.uniform(-0.03, 0.03, (n, k, d, 1))
    sv_rgb = (site_col[:, :, None, :] - 0.5 + sheen).reshape(n, k, 3 * d)

    ck = RawCheckpoint(
        points=P.astype(np.float32),
        radii=inverse_softplus(r).astype(np.float32),
        density=np.full(n, sigma, np.float32),  # softplus(300*100)/100 == 300 (linear regime)
        quaternions=quat.astype(np.float32),
        texel_sites=texel_sites.astype(np.float32),
        texel_sv_axis=sv_axis.astype(np.float32),
        texel_sv_rgb=sv_rgb.astype(np.float32),
        texel_height=texel_height.astype(np.float32),
        config={
            "dataset": "synthetic",
            "scene": "synth_room",
            "num_texel_sites": k,
            "sv_dof": d,
            "bkgd_color": [0.0, 0.0, 0.0],
            "is_pinhole": True,
            "synthetic": {"spacing": spacing, "seed": seed, "room_half": ROOM_HALF, "room_height": ROOM_H},
        },
    )
    kinds = np.concatenate([[kind] * p.shape[0] for p, _, kind in parts]).tolist()
    return ck, {"kinds": kinds}


def test_cameras(width=320, height=240, fov_deg=70.0) -> list[dict]:
    """Cameras in the repo's convention: right/up scaled by tan(half-fov), forward = up × right."""
    aspect = width / height
    ty = np.tan(np.radians(fov_deg) / 2)
    tx = ty * aspect
    cams = []
    specs = [
        ("corner", [-3.0, 1.6, 3.0], [1.0, -0.15, -1.0]),
        ("box", [0.0, 1.4, 0.5], [0.8, -0.3, -1.0]),
        ("sphere", [0.5, 1.2, 0.0], [-1.0, -0.2, 0.8]),
        ("pillar", [1.5, 1.7, 0.0], [-0.5, 0.0, -1.0]),
    ]
    for name, eye, fwd in specs:
        eye, fwd = np.array(eye), np.array(fwd, float)
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 1.0, 0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        assert np.allclose(np.cross(up, right), fwd)
        cams.append({"name": name, "eye": eye.tolist(), "right": (tx * right).tolist(), "up": (ty * up).tolist(), "width": width, "height": height})
    return cams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scenes/synth_room")
    ap.add_argument("--spacing", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--d", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ck, extra = make_scene(args.spacing, args.k, args.d, args.seed)
    out = Path(args.out)
    save_checkpoint(out, ck)
    (out / "cameras.json").write_text(json.dumps(test_cameras(), indent=1))
    print(f"wrote {out}/model.pt: {ck.n} cells, k={ck.k}, d={ck.d}")


if __name__ == "__main__":
    main()
