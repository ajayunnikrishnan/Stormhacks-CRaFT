# PROGRESS

## Phase 0 — Recon — DONE (commit a111efd)
`docs/ARCHITECTURE.md`; Power Foam pinned as submodule `powerfoam/` @ 9639225.

## Phase 1 — Flat port — DONE
Stack: TypeScript + WebGL2 + Vite (`web/`), Python 3.12 venv (`.venv/`) for tools.

- `tools/pf_common.py` — checkpoint I/O, activations, quaternion frames, Steiner points
  (port of `benchmark.py:get_steiner_points`), regular-triangulation adjacency (4D lift +
  qhull, port of `build_power_adjacency`), Morton sort, brute-force power-cell owner.
- `tools/synth_scene.py` — synthetic room in the repo's `model.pt` format (stand-in until a
  trained checkpoint arrives). 24.5k cells, k=4, D=4.
- `tools/export_scene.py` — checkpoint → `scene.json` + `scene.bin` (fp16 attrs, fp32
  positions, CSR adjacency). Drops redundant (empty power cell) sites. `--verify` checks
  adjacency against exact brute force.
- `tools/ref_render.py` — numpy port of `RayTracer.benchmark_kernel`, same order of ops.
- `web/` — scene loader → data textures (row-wrapped 4096-wide), SV pre-pass
  (`shaders/sv_prepass.frag`), flat walk (`shaders/walk_flat.frag`), composite, fly camera,
  HUD with FPS and GPU pass timers, `window.surveyor.{render,bench}` test hooks.
- `web/harness.html` — renders the manifest cameras and compares to the CPU reference.

### Acceptance (synthetic scene, 160×120, threshold 1e-2, near-cull on, repo pixel grid)
| camera | PSNR vs CPU reference | max abs err | px > 0.02 |
|---|---|---|---|
| corner | 72.1 dB | 0.008 | 0 |
| box | 73.6 dB | 0.006 | 0 |
| sphere | 73.5 dB | 0.004 | 0 |
| pillar | 72.7 dB | 0.009 | 0 |

Target was > 40 dB. **Pending:** the same comparison against the repo's actual Warp kernel
on a trained scene (needs a CUDA machine + checkpoint; `tools/ref_render.py` is the
line-by-line port, so this is a sanity check of the port's fidelity, not of our math).

### Performance (Apple M1 Pro, Chrome/ANGLE-Metal, 24.5k cells, walk + SV pre-pass, ms/frame)
| view | 1920×1080 | 1280×720 | 960×540 |
|---|---|---|---|
| corner (long rays, 16 cells/ray) | 47.1 | 23.0 | 14.3 |
| box | 20.1 | 10.1 | 6.7 |
| sphere | 21.8 | 10.9 | 7.0 |
| pillar | 24.2 | 12.1 | 7.5 |
SV pre-pass alone: 0.3–0.7 ms. CPU start-cell search: < 0.5 ms.
=> 30 fps at 1080p holds for most views on an integrated GPU; the worst view needs ~75% scale.
Dynamic resolution scaling is in (manual slider now; auto in Phase 8).

## Phase 2 — Geometry library — DONE
- `web/src/geometry/space.ts` and `web/shaders/geometry.glsl`, mirrored function by function
  with equation numbers: form (1), distance (2, chord formula), geodesic (3), transported
  direction (4), tangent toward (5), plane crossing (6) incl. S³ exit/entry periodicity,
  ball interval (7) for all κ, embedding (§3.6), isometries / inverse / J-Gram-Schmidt (§3.7).
- `web/tests/space.test.ts`: 27 vitest tests — on-model, Eq. 5 round trip, Eq. 6/7 vs
  numeric roots (bisection), flat limit O(s²), isometry MᵀJM=J + inverse + distance
  preservation, 10⁶-step bounded drift test (< 1e-5), transport composition.
- `web/src/geometry/probe.ts` + `shaders/geometry_probe.frag`: GLSL vs TS on 4096 random
  cases per κ per function; max rel err ≤ 2e-5 (fp32), 0 mismatches (see /harness.html).
- Finding worth writing up: an unbounded random walk in H³ drifts ~12 units in 10⁶ steps;
  there the hyperboloid coordinates reach ~1e5 and plain fp64 cancellation in ⟨x,x⟩ already
  exceeds 1e-5. The game keeps W bounded by re-centring on domain crossings, so this is a
  design constraint (keep the camera near the world origin), not a bug.
- Note for the writeup: "MᵀJM = J" characterises isometries only for κ ≠ 0; for E³
  (degenerate J) the condition is top row (1,0,0,0) + orthonormal rotation block.

## Next — Phase 3: curved scene embedding + curved walker (primary rays, single scene).

## Known issues
- No CUDA on the dev Mac; PSNR vs the real Warp kernel pending (needs lab GPU + checkpoint).
- Synthetic scene shows faint speckle on surfaces seen at grazing angles (synthetic radii
  slightly small for grazing coverage); cosmetic, will retune radii when visuals matter.
- Outer Steiner cells reach degree 547; costs the "corner" view. Consider capping Steiner
  radius ratio or an extra Steiner iteration.
- GPU timer queries (EXT_disjoint_timer_query_webgl2) report garbage when the tab is hidden;
  `bench()` uses a 1-px readback to sync instead.
- `{{TARGET_GPU_AND_OS}}`, `{{HOURS}}`, `{{TEAM_SIZE}}` still unknown.

## How to run
```
. .venv/bin/activate
python tools/synth_scene.py --out scenes/synth_room
python tools/export_scene.py scenes/synth_room --out web/public/scenes/synth_room --verify
python -m pytest tests/python -q
cd web && npm install && npx vite          # http://127.0.0.1:5173  (harness: /harness.html)
```
