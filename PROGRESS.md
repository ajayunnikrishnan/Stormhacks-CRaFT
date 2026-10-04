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

## Phase 4 — Topology — DONE (census manifolds via SnapPy deferred, cut-order item 1)
- `web/src/topology/domain.ts`: fundamental polyhedra as face covectors + pairing isometries;
  box domains (3-torus, half-turn space, non-orientable Klein space, {4,3,4}), cube honeycombs
  {4,3,5} (h = asinh√cos72° = 0.5306) and {4,3,3} tesseract (h = π/4) sized from the dihedral
  angle; `dodecahedron.ts`: Poincaré dodecahedral space (S³, 120°, 36° twist, h = π/10) and
  Seifert–Weber space (H³, 72°, 108° twist, h = 0.996). Vertices by triple plane intersection.
- Walker (`walk_curved.frag`): domain-face exit test per step, segment clipped at the face,
  ray state transported by g_f, re-projected, nudged inward; point location = per-face 32×32
  chart (exp-map coords at the face centre) → cell-id guess → steepest ascent on ⟨x,a⟩'.
  Hop cap 16, fog distance as the step cutoff (40 m default; S³ also stops at 2π).
- `locate.ts` builds the charts on the CPU with warm starts (3–6 ms per domain per κ).
- Camera: `IsoCamera.recentre()` applies Wb ← Wb·g_f⁻¹ when the eye leaves through f.
- UI: "space" picker; curved spaces lock k so the scene's horizontal extent fills the inradius.
- `tools/synth_scene.py --open` (floor + landmarks, no walls) so copies are visible;
  `export_scene.py --steiner-box` samples Steiner candidates uniformly in the padded bbox
  (the repo's Normal rule leaves the "sky" to a few huge cells: max degree 484 → 243, and
  tiled frames 3–7× faster).

### Tests (vitest, all green: 54 total)
| test | result |
|---|---|
| dihedral angles measured numerically at edges: {4,3,4}/{4,3,5}/{4,3,3} | 90.000 / 72.000 / 120.000° |
| dodecahedra: 20 vertices each on 3 faces, dihedral 120° (PDS) / 72° (SW) | pass |
| every g_f maps 50 sampled face points onto the partner plane; g_partner∘g_f = I | pass, 1e-9 |
| edge cycles compose to the identity: 3 per edge (PDS), 5 per edge (SW); wrong twists fail | pass |
| transported rays stay inside the domain over 20 hops (all 8 spaces) | pass |
| grid-located start == brute force on random face points (6 spaces × 900 pts) | 0 mismatches, ≤ 3 ascent steps |

### Performance (M1 Pro, 1280×720, open scene 9.7k cells, fog 30 m, hop cap 16, ms/frame)
| space | ms | | space | ms |
|---|---|---|---|---|
| none | 11.7 | | {4,3,5} | 42–45 |
| 3-torus | 44 | | tesseract {4,3,3} | 62 |
| Poincaré dodecahedral | 65 | | Seifert–Weber | 61 |
At 960×540: 29–40 ms. Sky rays dominate (they run to the fog); a closed scene is much cheaper.

## Phase 5 — Navigation & feedback — DONE
- `web/src/game/player.ts`: cell tracking by steepest ascent (no per-frame brute force);
  collision = geodesic probe of the destination + a ring at the player radius against dense
  foam (inside the cell's ball ∧ behind its dipole plane ∧ σ > 1e-3), sliding along x/z;
  compass = tangent vector in body coordinates: unchanged by translations (the frame and the
  vector are carried by the same transvection), rotated by every yaw ⇒ after a closed loop it
  is off the heading by the holonomy −κ·A.
- `web/src/ui/overlay.ts`: 2D overlay with the compass dial, the map inset (Poincaré ball top
  view for H³, stereographic for S³, metres for E³; domain edges as geodesic polylines, scene
  footprint, player + heading, beacons, laser) and the laser geodesic projected into the image
  through Eq. (5) tangents (overlay only, not occluded yet).
- `IsoCamera.setFromRepoCamera` now splits pitch from the body yaw, so walking stays on the eye
  plane; hooks route movement through the Player's collision and yaw through the compass.
- Keys: WASD/shift move, drag look, L laser, M map, B beacon (3 max). Scenes are exported with
  `--centre-y 1.6`, i.e. the eye plane is 1.6 m above the floor.

### Tests (vitest: 60 green)
| test | result |
|---|---|
| compass holonomy: geodesic triangle in E³/H³/S³, compass angle == 2π − Σ exterior == κ·Area (Gauss–Bonnet via Eq. 5 angles) | pass, < 2e-3 rad |
| camera import: W == R·T⁻¹ for exact repo cameras, body forward ⟂ transported vertical | pass, 2e-16 |
| in-browser: walk into the pillar from the "pillar" camera (flat and k = −0.05) | stops 0.64 m from the axis = 0.4 (pillar) + 0.25 (player) |
| in-browser: curved κ=0 vs flat after the camera refactor | 81–95 dB on 4 cameras |

## Phase 6 — Lighting & audio — DONE
- Eq. (8) `irradiance` in TS + GLSL (`irradianceK`): E = Φ/(4π)·max(0,⟨u,n⟩)/sn_κ(d)², with the S³
  long-way term (2π − d, −u). **Flux test** (vitest): Monte-Carlo over geodesic spheres of 4 radii
  per κ, E·4π sn²(R) = Φ to 1e-9; back faces get 0 direct and > 0 antipodal in S³.
- Pipeline: primary walk writes a G-buffer (MRT: baked rgb+T, median-depth hit X, transported
  normal, (t, cell, T_before, x₀)); `shadow.frag` walks geodesics from X to each of ≤ 4 lamps at
  half resolution accumulating transmittance (same Eq. 6/7 traversal; dense part = ball ∩ exact
  dipole half-space); `shade.frag`: C = baked ⊙ (ambient + Σ ρ·colour·(E·shadow·e^{−σd} +
  E_anti·e^{−σ(2π−d)})), flashlight = spotlight at the camera whose shadow is the primary ray's
  own transmittance, lamp glows by closest geodesic approach (closed form), fog exp(−σt) that
  doubles as the walk cutoff, Reinhard tone map.
- `lights.ts`: lamps (P: 1.5 m ahead) and flares (T: thrown along the view geodesic at 7 m/s,
  stop at dense foam via the player's collision probe, 12 s, flicker). F toggles the flashlight.
- `audio/curvedAudio.ts`: lamps hum (detuned sawtooth → lowpass), flares crackle (gated noise);
  gain ∝ 1/sn_κ(d)² normalised at 1.5 m (+ long way round in S³), panned by the Eq. 5 arrival
  direction in camera coordinates. Opt-in checkbox (browser autoplay policy).
- Approximation documented: lighting is evaluated once per pixel at the median-depth hit and
  multiplies the whole accumulated baked colour; semi-transparent edges inherit it.

### Measurements (M1 Pro, 1280×720, open scene, 2 lamps + flashlight, shadows at ½ res)
| | ms/frame |
|---|---|
| walk + shadows + shading, κ = 0 | 21.5 |
| walk + shadows + shading, k = −0.05 | 26.2 |
Baked-output parity (curved κ=0 vs flat) after the G-buffer refactor: 80.8 dB.

## Phase 7 — Gameplay — DONE
- `game/tools.ts`: beacon triangle (interior angles via Eq. 5 tangents, geodesic sides in metres;
  no area shown), light meter (scalar irradiance Φ/(4π sn²d) vs distance, flat 1/d² reference
  through the first sample), numeric Gauss–Bonnet area integration (for tests/writeup).
- `game/levels.ts`: 1 flat tutorial (beacons, meter, laser) · 2 "The lamps are dying" (H³,
  k = −0.12, 3 lamps, light the marker to 0.8 W/m²) · 3 "The light comes home" (S³, k = 0.042,
  marker 9 m out over the void; only a lamp near its antipode lights it) · 4 final exam (hidden
  weak k ∈ ±[0.006, 0.036], fog, submit on the slider with Enter → true k, error %, score, and a
  6 s morph flat-and-back) · sandbox with lesson cards per space. Tools are gated per level.
- `ui/hud.ts`: goal banner with live progress, triangle readout, meter plot, level banners,
  Esc pause menu with controls. In-view markers (beacons A/B/C, target ★, lamps) in the overlay.
- Player: floor support required (no walking off the platform); probe band covers the O(κx²)
  variation of the floor's geodesic depth.

### Tests
| test | result |
|---|---|
| flat triangle 3-4-5: angles 90/…, sum 180°, sides exact | pass |
| Gauss–Bonnet: numerically integrated area × κ == angle excess, H³ and S³, 3 sizes | pass, < 2e-3 |
| light meter ordering: H³ below 1/d², S³ above, E³ exact | pass |
| in-browser level flow: tutorial completes; exam with a 10 %-off guess scores 90; level 3: lamp at edge 0.026, at antipode 2.6e4, 0.5 m off 2.6, 1.5 m off 0.30 W/m² | pass |

## Phase 8 — Polish — DONE
- In-world face-pair tints (walker multiplies light through each face pair by the gallery colour,
  toggleable); topology gallery (G): per-space card with a gluing diagram generated at runtime
  from the domain's actual vertices/faces/pairings (twist angle or flip glyph on the visible face
  of each pair) and a thumbnail rendered by our own renderer; curvature badge, orientability, hint.
- Quality presets Low/Medium/High (render scale, shadow resolution, hop cap) and **auto** mode
  that adapts the render scale to hold 30 fps; FOV and mouse-sensitivity sliders; pause menu.
- Art: dark world, warm lamps, fog colour, Reinhard tone map, lamp glows (from Phase 6).
- Verified in-browser: all 9 cards have diagrams + thumbnails; 36°/108°/180° labels and the
  Klein flip glyph present; tints change the tiled render; auto quality settled at 0.6 scale /
  42 fps on the M1 Pro at 2048×1536.

## Phase 9 — Documentation & demo — DONE (video pending: needs a visible screen)
- `docs/WRITEUP.md`: models, linear-bisector argument, closed-form intersections, direction
  transport, falloff from sphere area, Gauss–Bonnet + holonomy, numerical findings, architecture,
  validation table, performance table, limitations (Killing–Hopf), prior art, the exact
  rasterisation bonus result; figures from `tools/make_figures.py` in `docs/figures/`.
- `docs/DEMO_SCRIPT.md` (2–3 min live script), `README.md` (run / deploy / checkpoint drop-in),
  `docs/ARCHITECTURE.md` as-built addendum, `.github/workflows/pages.yml` (tests + static build).
- Exported scene binaries are now tracked so the Pages build is self-contained (11 MB).

## Still open
- PSNR of the GPU walk against the repo's own Warp kernel on a **trained** scene: needs a CUDA
  machine and a checkpoint (`README.md` → "Using a trained Power Foam checkpoint").
- Cut per the prompt's cut order: level 5 (topology identification) and SnapPy census manifolds.
- Backup demo video: record `docs/DEMO_SCRIPT.md` on a machine with a visible browser.

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
