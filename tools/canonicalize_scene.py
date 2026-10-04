"""Put a trained Power Foam checkpoint into CRaFT's canonical frame: metres, y up, floor at y = 0,
origin under the camera centroid, and optionally cropped to a radius.

COLMAP scenes (Mip-NeRF 360 etc.) come in arbitrary units with an arbitrary orientation. The web
walker, the curvature slider (k in 1/m²), the player's eye height and the box universes all assume
metres and y-up, so we apply one similarity transform x' = s·(R x − c) to the WHOLE checkpoint and
save a new model.pt. Everything the renderer evaluates is covariant under a similarity:

  points     p' = s (R p − c)
  radii      r' = s r                       (raw radii pass through softplus: raw' = softplus⁻¹(s r))
  density    σ' = σ / s                     (transmittance exp(−σ t) is preserved: t' = s t)
  frames     (n, t, b)' = (R n, R t, R b)   → quaternion
  sites      texel_sites, texel_height are in units of r → unchanged
  SV axes    a' = R a                       (view-direction basis rotates with the scene)
  cameras    eye' = s (R eye − c), right' = R right, up' = R up

Estimation (needs cameras.json: [{eye, right, up}] — the training cameras dumped by
tools/render_reference.py --dump-cameras):
  up      = normalised mean of the cameras' up vectors (handheld captures hold the camera level)
  floor   = low percentile of the heights of dense cells under the cameras
  scale   = assumed camera height above the floor (default 1.5 m) / measured height in scene units
  centre  = camera centroid horizontally, floor vertically
  yaw     = first camera's horizontal forward → −z

Usage:
    python tools/canonicalize_scene.py scenes/treehill_raw --out scenes/treehill \
        --cam-height 1.5 --crop-radius 12 --crop-below 1.5 --crop-above 12
Writes model.pt, config.yaml, cameras.json (transformed), frame.json (the transform) and copies refs/.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from pf_common import RawCheckpoint, frame_to_quat, inverse_softplus, load_checkpoint, quat_frame, save_checkpoint, softplus


def rotation_from_to(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a to unit vector b (Rodrigues)."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0: return np.eye(3)
        # 180°: rotate about any axis orthogonal to a
        o = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = np.cross(a, o); ax /= np.linalg.norm(ax)
        return 2 * np.outer(ax, ax) - np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def rot_y(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def transform_checkpoint(ck: RawCheckpoint, R: np.ndarray, s: float, c: np.ndarray) -> RawCheckpoint:
    """Apply x' = s (R x − c) to every attribute (see module docstring)."""
    R = np.asarray(R, np.float64); c = np.asarray(c, np.float64)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0, "R must be a proper rotation"
    pts = (s * (ck.points.astype(np.float64) @ R.T - c)).astype(np.float32)
    # clamp: softplus underflows to exactly 0 for very negative raw values, and softplus⁻¹(0) = −∞
    radii = inverse_softplus(np.maximum(s * softplus(ck.radii.astype(np.float64)), 1e-9)).astype(np.float32)
    density = inverse_softplus(np.maximum(softplus(ck.density.astype(np.float64)) / s, 1e-9)).astype(np.float32)
    n, t, b = quat_frame(ck.quaternions.astype(np.float64))
    q = frame_to_quat(n @ R.T, t @ R.T, b @ R.T).astype(np.float32)
    N, k = ck.n, ck.k
    axes = ck.texel_sv_axis.astype(np.float64).reshape(N, k, -1, 3) @ R.T
    return RawCheckpoint(
        points=pts, radii=radii, density=density, quaternions=q,
        texel_sites=ck.texel_sites.copy(), texel_sv_axis=axes.reshape(N, k, -1).astype(np.float32),
        texel_sv_rgb=ck.texel_sv_rgb.copy(), texel_height=ck.texel_height.copy(), config=dict(ck.config),
    )


def transform_cameras(cams: list[dict], R: np.ndarray, s: float, c: np.ndarray) -> list[dict]:
    out = []
    for cam in cams:
        d = dict(cam)
        d["eye"] = (s * (R @ np.asarray(cam["eye"], np.float64) - c)).tolist()
        d["right"] = (R @ np.asarray(cam["right"], np.float64)).tolist()
        d["up"] = (R @ np.asarray(cam["up"], np.float64)).tolist()
        out.append(d)
    return out


def estimate_frame(ck: RawCheckpoint, cams: list[dict], cam_height_m: float, floor_pct: float = 3.0, min_opacity: float = 0.5) -> dict:
    eyes = np.array([c["eye"] for c in cams], np.float64)
    ups = np.array([c["up"] for c in cams], np.float64)
    pts = ck.points.astype(np.float64)
    r = softplus(ck.radii.astype(np.float64)); sig = softplus(ck.density.astype(np.float64))
    opaque = 1 - np.exp(-sig * 2 * r) > min_opacity  # optical depth across the cell
    up = ups.mean(0); up /= np.linalg.norm(up)  # stage 1: cameras are held roughly level

    def analyse(up):
        R1 = rotation_from_to(up, np.array([0, 1.0, 0]))
        E = eyes @ R1.T
        centre_xz = E[:, [0, 2]].mean(0)
        orbit = float(np.max(np.linalg.norm(E[:, [0, 2]] - centre_xz, axis=1)))
        P = pts @ R1.T
        near = np.linalg.norm(P[:, [0, 2]] - centre_xz, axis=1) < 1.5 * orbit
        sel = opaque & near
        if sel.sum() < 100: sel = near
        # Surface cells have their centre ON the surface (dipole model), so the floor is the median
        # centre height of the lowest layer: cells within two median radii above a low percentile
        # (the percentile alone would follow floaters under the ground in real captures).
        y = P[:, 1]
        f_lo = float(np.percentile(y[sel], floor_pct))
        rm = float(np.median(r[sel]))
        layer = sel & (y >= f_lo - rm) & (y <= f_lo + 2 * rm)
        floor_u = float(np.median(y[layer])) if layer.sum() >= 20 else f_lo
        cam_h_u = float(E[:, 1].mean() - floor_u)
        floor_cells = sel & (y < floor_u + 0.15 * cam_h_u) & (y > floor_u - 0.15 * cam_h_u)
        return R1, E, centre_xz, orbit, floor_u, cam_h_u, floor_cells

    R1, E, centre_xz, orbit, floor_u, cam_h_u, floor_cells = analyse(up)
    n_floor = int(floor_cells.sum())
    if n_floor >= 50:
        # stage 2: refine up as the normal of the plane through the floor cells (handheld cameras tilt a few degrees)
        Q = pts[floor_cells]; Q = Q - Q.mean(0)
        nrm = np.linalg.svd(Q, full_matrices=False)[2][-1]
        if nrm @ up < 0: nrm = -nrm
        up = nrm / np.linalg.norm(nrm)
        R1, E, centre_xz, orbit, floor_u, cam_h_u, floor_cells = analyse(up)
    # yaw: first camera's horizontal forward → −z
    f0 = np.cross(np.asarray(cams[0]["up"]), np.asarray(cams[0]["right"]))
    f0 = R1 @ f0; f0[1] = 0
    yaw = 0.0 if np.linalg.norm(f0) < 1e-9 else float(np.arctan2(f0[0], -f0[2]))
    R = rot_y(-yaw) @ R1
    E = eyes @ R.T
    centre_xz = E[:, [0, 2]].mean(0)
    s = cam_height_m / cam_h_u
    c = np.array([centre_xz[0], floor_u, centre_xz[1]])
    return {"R": R.tolist(), "scale": s, "centre": c.tolist(), "up_scene": up.tolist(), "yaw_deg": float(np.degrees(yaw)),
            "floor_scene_units": floor_u, "camera_height_scene_units": cam_h_u, "camera_height_m": cam_height_m,
            "orbit_radius_m": orbit * s, "n_cameras": len(cams), "n_floor_cells": int(floor_cells.sum())}


def crop(ck: RawCheckpoint, radius: float | None, below: float | None, above: float | None) -> tuple[RawCheckpoint, np.ndarray]:
    P = ck.points
    keep = np.ones(ck.n, bool)
    if radius is not None: keep &= np.hypot(P[:, 0], P[:, 2]) <= radius
    if below is not None: keep &= P[:, 1] >= -below
    if above is not None: keep &= P[:, 1] <= above
    idx = np.flatnonzero(keep)
    return RawCheckpoint(points=P[idx], radii=ck.radii[idx], density=ck.density[idx], quaternions=ck.quaternions[idx],
                         texel_sites=ck.texel_sites[idx], texel_sv_axis=ck.texel_sv_axis[idx], texel_sv_rgb=ck.texel_sv_rgb[idx],
                         texel_height=ck.texel_height[idx], config=ck.config), idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir"); ap.add_argument("--out", required=True)
    ap.add_argument("--cameras", default=None, help="cameras.json (default: <scene_dir>/cameras.json or refs/cameras.json)")
    ap.add_argument("--cam-height", type=float, default=1.5, help="assumed mean camera height above the floor, metres")
    ap.add_argument("--crop-radius", type=float, default=None, help="drop cells farther than this (m) horizontally from the centre")
    ap.add_argument("--crop-below", type=float, default=None, help="drop cells more than this (m) below the floor")
    ap.add_argument("--crop-above", type=float, default=None, help="drop cells more than this (m) above the floor")
    ap.add_argument("--floor-pct", type=float, default=3.0)
    args = ap.parse_args()
    sd, out = Path(args.scene_dir), Path(args.out)
    ck = load_checkpoint(sd)
    cam_path = Path(args.cameras) if args.cameras else (sd / "cameras.json" if (sd / "cameras.json").exists() else sd / "refs" / "cameras.json")
    cams = json.loads(cam_path.read_text())
    fr = estimate_frame(ck, cams, args.cam_height, args.floor_pct)
    R, s, c = np.array(fr["R"]), fr["scale"], np.array(fr["centre"])
    ck2 = transform_checkpoint(ck, R, s, c)
    n0 = ck2.n
    ck2, idx = crop(ck2, args.crop_radius, args.crop_below, args.crop_above)
    fr.update(n_cells_before_crop=int(n0), n_cells_after_crop=int(ck2.n), crop={"radius": args.crop_radius, "below": args.crop_below, "above": args.crop_above})
    ck2.config = dict(ck2.config); ck2.config["craft_canonical"] = {k: v for k, v in fr.items() if k != "R"}
    save_checkpoint(out, ck2)
    # the scene's visible cameras: the rendered reference cameras when there are any (the harness
    # compares camera i against refs/cam_i), else the estimation cameras
    vis = json.loads((sd / "refs" / "cameras.json").read_text()) if (sd / "refs" / "cameras.json").exists() else cams
    (out / "cameras.json").write_text(json.dumps(transform_cameras(vis, R, s, c), indent=1))
    (out / "frame.json").write_text(json.dumps(fr, indent=1))
    if (sd / "refs").exists():
        shutil.copytree(sd / "refs", out / "refs", dirs_exist_ok=True)
        (out / "refs" / "cameras.json").write_text(json.dumps(transform_cameras(json.loads((sd / "refs" / "cameras.json").read_text()), R, s, c), indent=1))
    P = ck2.points
    print(json.dumps({k: v for k, v in fr.items() if k != "R"}, indent=1))
    print("bbox (m):", P.min(0).round(2).tolist(), P.max(0).round(2).tolist())


if __name__ == "__main__":
    main()
