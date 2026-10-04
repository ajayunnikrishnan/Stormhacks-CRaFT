"""Export a Power Foam checkpoint to the compact binary the web renderer loads.

Pipeline (docs/ARCHITECTURE.md §11):
  1. load model.pt + config.yaml, apply activations, derive frames
  2. Steiner points (benchmark.py port) so empty space has well-shaped cells
  3. Morton sort for texture-cache locality
  4. regular-triangulation adjacency (dual of the unbounded power diagram)
  5. optional brute-force verification
  6. write scene.json (manifest) + scene.bin (sections)

Binary layout (little endian, sections 16-byte aligned, offsets in manifest):
  pos      f32 [N,4]   xyz, radius            -> RGBA32F texture, N texels
  nsigma   f16 [N,4]   normal xyz, sigma      -> RGBA16F, N
  siteoff  f16 [N*k,4] offset xyz, height     -> RGBA16F, N*k
  svaxis   f16 [N*k*D,3] raw axis (|axis| = temperature) -> RGB16F
  svrgb    f16 [N*k*D,3]                                  -> RGB16F
  adjoff   u32 [N+1]                          -> R32UI
  adjidx   u32 [E]                            -> R32UI

Usage:
    python tools/export_scene.py scenes/synth_room --out web/public/scenes/synth_room [--no-steiner] [--verify]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pf_common import (
    FoamScene,
    activate,
    append_steiner,
    brute_force_power_cell,
    build_power_adjacency,
    load_checkpoint,
    morton_order,
    permute_scene,
    steiner_points,
)


def prepare_scene(scene_dir: Path, steiner: bool = True, seed: int = 0) -> tuple[FoamScene, dict]:
    t0 = time.time()
    ck = load_checkpoint(scene_dir)
    s = activate(ck)
    n_scene = s.n
    info = {"n_scene_cells": int(n_scene), "k": int(s.k), "d": int(s.d)}
    if steiner:
        st_pos, st_r = steiner_points(s.pos, s.radius, seed=seed)
        s = append_steiner(s, st_pos, st_r)
        info["n_steiner_cells"] = int(st_pos.shape[0])
    else:
        info["n_steiner_cells"] = 0
    perm = morton_order(s.pos)
    s = permute_scene(s, perm)
    s.adj_offsets, s.adj_index = build_power_adjacency(s.pos, s.radius)
    deg = np.diff(s.adj_offsets)
    # Sites with an empty power cell (hidden by neighbours' weights) get no edges
    # in the regular triangulation and can never be visited; drop and re-index.
    # Removing a redundant site does not change the power diagram of the others.
    redundant = deg == 0
    info["n_redundant_dropped"] = int(redundant.sum())
    if redundant.any():
        keep = np.flatnonzero(~redundant)
        remap = -np.ones(s.n, np.int64)
        remap[keep] = np.arange(keep.size)
        n_scene_kept = int((keep < n_scene).sum())
        s = permute_scene(s, keep)
        s.adj_offsets, s.adj_index = build_power_adjacency(s.pos, s.radius)
        deg = np.diff(s.adj_offsets)
        assert (deg > 0).all(), "redundant removal must be idempotent"
        info["n_scene_cells"] = n_scene_kept
    info.update(
        n_cells=int(s.n),
        n_edges=int(s.adj_index.shape[0] // 2),
        avg_degree=float(deg.mean()),
        max_degree=int(deg.max()),
        prep_seconds=round(time.time() - t0, 2),
    )
    return s, info


def verify_adjacency(s: FoamScene, n_samples: int = 20000, seed: int = 0) -> dict:
    """Check that greedy 'exit through first bisector' walks agree with brute-force
    power-cell ownership: for random x, the owner must be reachable by descending
    power distance over adjacency from the nearest-site start, and the owner's
    power must beat every neighbour's (local optimality == global optimality for
    a correct regular triangulation)."""
    rng = np.random.default_rng(seed)
    lo, hi = s.pos.min(0), s.pos.max(0)
    x = rng.uniform(lo, hi, (n_samples, 3))
    owner = _exact_power_owner(s, x)
    # local optimality test at the owner
    bad = 0
    for i in range(n_samples):
        o = owner[i]
        nb = s.adj_index[s.adj_offsets[o] : s.adj_offsets[o + 1]]
        pw_o = np.sum((x[i] - s.pos[o]) ** 2) - s.radius[o] ** 2
        pw_n = np.sum((x[i] - s.pos[nb]) ** 2, -1) - s.radius[nb] ** 2
        if np.any(pw_n < pw_o - 1e-9):
            bad += 1
    # descent-walk test from a random start
    fails = 0
    for i in range(min(n_samples, 2000)):
        c = int(rng.integers(0, s.n))
        for _ in range(100000):
            nb = s.adj_index[s.adj_offsets[c] : s.adj_offsets[c + 1]]
            pw_c = np.sum((x[i] - s.pos[c]) ** 2) - s.radius[c] ** 2
            pw_n = np.sum((x[i] - s.pos[nb]) ** 2, -1) - s.radius[nb] ** 2
            j = int(np.argmin(pw_n))
            if pw_n[j] < pw_c - 1e-12:
                c = int(nb[j])
            else:
                break
        if c != owner[i]:
            fails += 1
    return {"samples": n_samples, "local_optimality_violations": bad, "descent_walk_mismatches": fails}


def _exact_power_owner(s: FoamScene, x: np.ndarray, chunk: int = 2048) -> np.ndarray:
    out = np.empty(x.shape[0], np.int64)
    r2 = s.radius**2
    for a in range(0, x.shape[0], chunk):
        xb = x[a : a + chunk]
        d2 = np.sum(xb[:, None, :] ** 2, -1) - 2 * xb @ s.pos.T + np.sum(s.pos**2, -1)[None, :]
        out[a : a + chunk] = np.argmin(d2 - r2[None, :], axis=1)
    return out


def write_scene(s: FoamScene, out_dir: Path, info: dict, extra_meta: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n, k, d = s.n, s.k, s.d
    sections = {}
    blobs = []
    offset = 0

    def add(name, arr, dtype, shape):
        nonlocal offset
        a = np.ascontiguousarray(arr.astype(dtype)).reshape(-1)
        pad = (-a.nbytes) % 16
        sections[name] = {"offset": offset, "dtype": np.dtype(dtype).name, "shape": list(shape), "nbytes": int(a.nbytes)}
        blobs.append(a.tobytes())
        if pad:
            blobs.append(b"\0" * pad)
        offset += a.nbytes + pad

    add("pos", np.concatenate([s.pos, s.radius[:, None]], -1), np.float32, (n, 4))
    add("nsigma", np.concatenate([s.normal, s.sigma[:, None]], -1), np.float16, (n, 4))
    add("siteoff", np.concatenate([s.site_off, s.height[..., None]], -1), np.float16, (n * k, 4))
    add("svaxis", s.sv_axis, np.float16, (n * k * d, 3))
    add("svrgb", s.sv_rgb, np.float16, (n * k * d, 3))
    add("adjoff", s.adj_offsets, np.uint32, (n + 1,))
    add("adjidx", s.adj_index, np.uint32, (s.adj_index.shape[0],))
    data = b"".join(blobs)
    (out_dir / "scene.bin").write_bytes(data)
    lo, hi = s.pos[: info["n_scene_cells"]].min(0), s.pos[: info["n_scene_cells"]].max(0)
    manifest = {
        "format": "surveyor-foam-v1",
        "n": n,
        "k": k,
        "d": d,
        "n_edges_directed": int(s.adj_index.shape[0]),
        "bbox_min": lo.tolist(),
        "bbox_max": hi.tolist(),
        "bin": "scene.bin",
        "bin_bytes": len(data),
        "sections": sections,
        "info": info,
    }
    if extra_meta:
        manifest.update(extra_meta)
    (out_dir / "scene.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-steiner", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    scene_dir = Path(args.scene_dir)
    s, info = prepare_scene(scene_dir, steiner=not args.no_steiner, seed=args.seed)
    if args.verify:
        info["verify"] = verify_adjacency(s)
        print("verify:", info["verify"])
    extra = {}
    cams = scene_dir / "cameras.json"
    if cams.exists():
        extra["cameras"] = json.loads(cams.read_text())
    m = write_scene(s, Path(args.out), info, extra)
    print(json.dumps({k: v for k, v in m.items() if k != "sections"}, indent=1))


if __name__ == "__main__":
    main()
