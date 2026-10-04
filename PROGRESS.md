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

## Phase 3 — Curved scene + curved walker (primary rays) — DONE
- `tools/curved.py`: embedding (§3.6), curved power-cell rule a_i = p_i/cs_κ(r_i) with the unified
  form ⟨x,a⟩' (κ'=1 for E³ with a=(−pm, p̄)), adjacency per κ (H³: Klein-model regular
  triangulation with c=ā/2, w=|c|²+κa₀; S³: 4D hull of {a_i}; E³: flat), κ-sweep union,
  exact CPU walk. `tools/export_scene.py --curved` ships the union graph + sweep stats.
- `tests/python/test_curved.py` (10): flat limit of cosh d/cosh r and cos d/cos r (O(s⁴)),
  cell rule == natural forms, bisector linearity, adjacency vs brute force at 5 curvatures
  (all three geometries), union-graph walk == exact walk at 4 curvatures NOT in the sweep.
- Union inflation on the synth room: **+15 % edges** (avg degree 13.1 → 15.1) over
  k ∈ [−0.14, +0.035] 1/m² (12 samples per sign; S³ limit = hemisphere with margin).
- `web/shaders/walk_curved.frag` + `sv_prepass_curved.frag`: exact traversal (Eqs. 6–7),
  locally-flat shading in the cell frame (§3.11), transported SV direction (Eq. 4).
  Compiled per κ (`#define KAPPA`) so branches fold. `web/src/foam/curved.ts` recomputes
  a_i, (R, cs R) on the CPU when the slider moves (one texSubImage2D each).
- `web/src/game/isocamera.ts`: camera at the origin, world→camera isometry W = R_pitch·W_body,
  body confined to the eye plane x₂ = 0, re-orthonormalised every frame; curvature changes
  keep the physical pose (metres) and yaw.
- `tools/ref_render_curved.py`: fp64 CPU mirror of the curved walker.
- **Numerical finding (writeup):** the hyperboloid formulation loses fp32 digits as s → 0
  (bisector offsets and the ball test are O(s²) differences of O(1) numbers). Fixed by
  storing a₀−1 and rewriting Eq. (7) in terms of the small chord ⟨o−p,o−p⟩ (same equations;
  see space.ts / geometry.glsl comments). The E³ switch is at |k| < 1e-9; at |k| = 1e-6 the
  curved and flat images agree to 43–52 dB, so the switch is invisible.

### Acceptance (synthetic room, camera "corner" unless noted)
| test | result |
|---|---|
| curved shader at κ=0 vs flat shader (4 cameras, 160×120) | 82.3 / 88.6 / 85.6 / 90.5 dB |
| curved vs flat at k = ±1e-6 (E³-switch continuity) | 51.8 / 42.8 dB, ≤ 32 px off by > 0.02 |
| GPU fp32 vs fp64 CPU reference, k = −0.12 (H³) | 45.6 dB, 26 / 6912 px > 0.02 |
| GPU fp32 vs fp64 CPU reference, k = +0.03 (S³) | 46.8 dB, 36 / 6912 px > 0.02 |
| GPU fp32 vs fp64 CPU reference, k = 0.001 | 66.2 dB |
| local-flat shading error max |Δt|/R (H³ k=−0.12 / S³ k=0.03) | 2.8e-3 / 6.8e-4 = 2.7–2.8 · R² |
| GLSL vs TS probe after the conditioning rewrite | 0 mismatches, max rel err 2e-5 |

### Performance, curved walker (M1 Pro, ms/frame, pre-pass + walk)
| view | k | 1920×1080 | 1280×720 |
|---|---|---|---|
| corner | 0 (E³ path) | 61.0 | 30.0 |
| corner | −0.12 (H³) | 53.7 | 25.5 |
| corner | +0.03 (S³) | 96.9 | 46.8 |
| pillar | 0 / −0.12 / +0.03 | 27.6 / 31.5 / 42.8 | 13.6 / 15.4 / 21.1 |
SV pre-pass: ~0.3 ms. The S³ variant pays for atan + periodic bookkeeping per neighbour.
30 fps at 1080p holds for typical views; the worst S³ view needs ~60 % render scale on this
integrated GPU (auto-scaling lands in Phase 8). Flat shader for reference: 47 ms / 23 ms.

## Next — Phase 4: topology (3-torus → {4,3,5} → S³ → closed manifolds).

## Known issues
- Curved walker ~1.3× slower than the flat shader on the E³ path (generic 4D math); S³ ~2×.
  Candidate optimisations: per-frame per-edge pre-pass of A = w·o; fewer transcendental calls.
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
