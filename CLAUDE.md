# CLAUDE.md — CRaFT

Hackathon entry (Huawei "Beyond Euclid"): **CRaFT, Curved Radiance Foam Tracing** — *Exact Geodesic
Ray Tracing of Radiance Foams in Constant-Curvature Spaces*. A WebGL2 first-person explorer that walks
a captured Power Foam scene along exact geodesics of H³/E³/S³, with a curvature slider, a guess-the-universe quiz, a gallery of
closed universes and a map inset. (The earlier game layer — beacons, lamps, light meter, laser,
levels — was removed at the user's request on 2026-10-04; the maths lives on in `game/tools.ts` and
the lighting passes stay in the renderer behind the developer panel.) Read `PROGRESS.md` first for
status, then `docs/WRITEUP.md` for the maths. Equation numbers in code comments (`// Eq. (6)`)
refer to the writeup.

## Layout

- `powerfoam/` — upstream repo as a git submodule. **Never modify it.** Export code lives in `tools/`.
- `tools/` — Python 3.12 (`.venv/`): `pf_common.py` (checkpoint I/O, Steiner, triangulation),
  `curved.py` (curved power diagram, κ sweep), `synth_scene.py`, `export_scene.py`,
  `ref_render.py` (numpy port of the Warp kernel), `ref_render_curved.py` (fp64 mirror of the
  curved shader), `render_reference.py` (runs INSIDE the Power Foam env on CUDA), `make_figures.py`,
  `render_polyhedra.py` (matplotlib 3/4-view pictures of every fundamental polyhedron for the
  gallery, from `web/tools/domains.json` which `node web/tools/dump_domains.mjs` writes from the TS
  domain definitions; output `web/public/thumbs/*.png`). Re-run both after changing `topology/domain.ts`.
- `web/` — Vite + TypeScript + WebGL2. `src/geometry/space.ts` ⇄ `shaders/geometry.glsl` are
  **mirrored line for line**; change both or neither. `src/topology/` domains + point location,
  `src/render/` passes (sv_prepass → walk_curved MRT G-buffer → shadow → shade), `src/game/`
  camera/player/tools, `src/ui/` map inset (overlay.ts) / gallery.
- `scenes/` — checkpoints in Power Foam's own format (`model.pt`, `config.yaml`, `cameras.json`).
  Exported scenes go to `web/public/scenes/<name>/` (tracked, so the static build is self-contained).
- `tests/python/` (pytest) and `web/tests/` (vitest). `web/harness.html` compares GPU vs CPU
  references and GLSL vs TS.

## Commands

```bash
. .venv/bin/activate && python -m pytest tests/python -q          # 22 tests
cd web && npx tsc --noEmit && npx vitest run                       # 67 tests
cd web && npx vite                                                 # http://127.0.0.1:5173 (?mode=explore|quiz|gallery)
python tools/synth_scene.py --out scenes/synth_open --open
python tools/export_scene.py scenes/synth_open --out web/public/scenes/synth_open --curved --sweep 12 --kmax 0.042 --steiner-box 2.0 --steiner-iters 14 --centre-y 1.6
```

Re-export BOTH `synth_open` (default scene) and `synth_room` after exporter changes.

## Rules that have bitten us

- **Never change the math silently.** New formulas get a test against numeric ground truth
  (bisection, brute force, Monte-Carlo) before the shader uses them. If a test disagrees with the
  spec, stop and report; don't tune constants.
- Keep the repo kernel's order of operations in `walk_flat.frag` / `ref_render.py`; conditioning
  rewrites (same equations, different order) go in the curved path and are documented in the
  writeup §8.
- Every shader that includes `common.glsl` gets both sampler precisions from it; don't redeclare.
- Curved shaders are compiled per κ via `#define KAPPA`; `k` must stay a local derived from it.
- The exporter must compute the scene bbox BEFORE Steiner points and the Morton sort.
- Performance knobs live in the main controls card: render distance (walk cutoff + unlit fog window,
  default 14 m) and detail chips (resolution presets; `auto` targets 30 fps). The canvas is 1× by
  default; HiDPI is a developer-panel option. The ray walk cost is roughly linear in render distance.
- Positions stored in the manifest/charts are metres in the scene frame; model coordinates are
  metres × s with the centre at the eye plane (`--centre-y 1.6`). Convert with `embedPoint` /
  `logAtOrigin`, never by scaling model vectors directly.
- Box universes (3-torus, half-turn, Klein) are sized from the scene bbox but must CONTAIN the
  floor slab: `domHalf` in main.ts puts the bottom face 5 cm below the floor surface. A face on or
  above the floor makes downward rays teleport forever (black lower half of the view).
- The Bash tool's cwd resets between calls: use absolute paths or `cd` inside each command.
- The browser pane may be hidden: verify through `window.craft.*` hooks (`render`, `renderCurved`,
  `renderLit`, `walk`, `setPose`, `start`, `setTopology`, `bench`, `gallery`) and PNG round-trips.
  `gl.finish()` does not block when hidden; benchmarks sync with a 1-px readback.

## Open items

- PSNR against the real Warp kernel on a **trained** scene (needs CUDA; see `docs/TRAINING_VASTAI.md`,
  then `tools/render_reference.py`). Everything so far used synthetic scenes.
- Cut per the prompt's cut order: level 5 (topology identification), SnapPy census manifolds.
- Demo video; GitHub remote + Pages deploy (`.github/workflows/pages.yml` is ready).
