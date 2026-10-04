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

import curved as cv
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


def prepare_scene(scene_dir: Path, steiner: bool = True, seed: int = 0, steiner_box_pad: float | None = None, steiner_iters: int = 10) -> tuple[FoamScene, dict]:
    t0 = time.time()
    ck = load_checkpoint(scene_dir)
    s = activate(ck)
    n_scene = s.n
    # scene-only bounding box, BEFORE Steiner points are appended and before the Morton sort
    # permutes cells (the first n_scene entries are not the scene cells afterwards)
    info = {"n_scene_cells": int(n_scene), "k": int(s.k), "d": int(s.d), "scene_bbox": [s.pos.min(0).tolist(), s.pos.max(0).tolist()]}
    if steiner:
        box = None
        if steiner_box_pad is not None:
            lo, hi = s.pos.min(0) - steiner_box_pad, s.pos.max(0) + steiner_box_pad
            box = (lo, hi)
            info["steiner_box"] = [lo.tolist(), hi.tolist()]
        st_pos, st_r = steiner_points(s.pos, s.radius, iterations=steiner_iters, seed=seed, sample_box=box)
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


def curved_union(s: FoamScene, scene_bbox: list, k_max: float | None, n_per_sign: int, verify_samples: int = 0, k_neg: float | None = None, centre_y: float | None = None) -> dict:
    """§3.5: union adjacency over a sweep of curvatures k ∈ [−k_neg, k_max] (1/m²).
    Default k_max keeps the whole scene inside an S³ hemisphere with margin:
    s_max · max|x − centre| = 1.2  (< π/2). H³ has no such limit; default k_neg = 4·k_max."""
    centre = 0.5 * (np.array(scene_bbox[0]) + np.array(scene_bbox[1]))
    if centre_y is not None:
        centre[1] = centre_y  # eye plane: the player walks on the totally geodesic plane through the centre
    ext = float(np.linalg.norm(s.pos - centre, axis=-1).max())
    if k_max is None:
        k_max = (1.2 / ext) ** 2
    if k_neg is None:
        k_neg = 4.0 * k_max
    ks = [-v for v in reversed(cv.sweep_values(k_neg, n_per_sign)[n_per_sign + 1 :])] + cv.sweep_values(k_max, n_per_sign)[n_per_sign:]
    off, idx, info = cv.union_adjacency(s, centre, ks, verify_samples=verify_samples)
    return {"centre": centre.tolist(), "k_max": k_max, "k_neg": k_neg, "scene_extent": ext, "offsets": off, "index": idx, "info": info}


def write_scene(s: FoamScene, out_dir: Path, info: dict, extra_meta: dict | None = None, union: dict | None = None) -> dict:
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
    if union is not None:
        add("adjoff_u", union["offsets"], np.uint32, (n + 1,))
        add("adjidx_u", union["index"], np.uint32, (union["index"].shape[0],))
    data = b"".join(blobs)
    (out_dir / "scene.bin").write_bytes(data)
    lo, hi = np.array(info["scene_bbox"][0]), np.array(info["scene_bbox"][1])
    manifest = {
        "format": "craft-foam-v1",
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
    if union is not None:
        manifest["curved"] = {
            "centre": union["centre"],
            "k_max": union["k_max"],
            "k_neg": union["k_neg"],
            "scene_extent": union["scene_extent"],
            "n_edges_directed_union": int(union["index"].shape[0]),
            "sweep": union["info"],
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
    ap.add_argument("--curved", action="store_true", help="also export the union adjacency over a curvature sweep (§3.5)")
    ap.add_argument("--kmax", type=float, default=None, help="max |k| in 1/m² (default: S³ hemisphere limit)")
    ap.add_argument("--sweep", type=int, default=16, help="sweep samples per sign")
    ap.add_argument("--kneg", type=float, default=None, help="max |k| on the H³ side (default 4·kmax)")
    ap.add_argument("--steiner-box", type=float, default=None, help="sample Steiner candidates uniformly in the scene bbox padded by this many metres (fills a domain around open scenes)")
    ap.add_argument("--steiner-iters", type=int, default=10)
    ap.add_argument("--centre-y", type=float, default=None, help="y (metres) of the embedding centre = the player's eye plane (e.g. floor + 1.6)")
    args = ap.parse_args()
    scene_dir = Path(args.scene_dir)
    s, info = prepare_scene(scene_dir, steiner=not args.no_steiner, seed=args.seed, steiner_box_pad=args.steiner_box, steiner_iters=args.steiner_iters)
    if args.verify:
        info["verify"] = verify_adjacency(s)
        print("verify:", info["verify"])
    extra = {}
    cams = scene_dir / "cameras.json"
    if cams.exists():
        extra["cameras"] = json.loads(cams.read_text())
    union = None
    if args.curved:
        union = curved_union(s, info["scene_bbox"], args.kmax, args.sweep, verify_samples=4000 if args.verify else 0, k_neg=args.kneg, centre_y=args.centre_y)
        ui = union["info"]
        print(f"curved sweep: k in [-{union['k_neg']:.4f}, {union['k_max']:.4f}]  union edges {ui['union_edges']}  inflation vs flat {ui['inflation_vs_flat']:.3f}  avg deg {ui['union_avg_degree']:.1f}  max deg {ui['union_max_degree']}")
        for st in ui["samples"]:
            print(f"  k={st['k']:+.4f} κ={st['kappa']:+d} s={st['s']:.3f} edges={st['n_edges']} avgdeg={st['avg_degree']:.1f} zero-deg={st['n_zero_degree']}" + (f" verify={st['verify']}" if "verify" in st else ""))
    m = write_scene(s, Path(args.out), info, extra, union)
    print(json.dumps({k: v for k, v in m.items() if k != "sections"}, indent=1))


if __name__ == "__main__":
    main()
