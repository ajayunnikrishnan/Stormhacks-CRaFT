# CRaFT: Curved Radiance Foam Tracing

*Exact Geodesic Ray Tracing of Radiance Foams in Constant-Curvature Spaces*

Hackathon entry for Huawei Custom Challenge #1, *Beyond Euclid*. A captured [Power Foam](https://github.com/theialab/powerfoam)
scene is ray traced in WebGL2 along exact geodesics of hyperbolic, flat and spherical space, with a
continuous curvature slider, a guess-the-universe quiz and a gallery of closed 3-manifolds (3-torus, half-turn, Klein,
{4,3,5}, tesseract, Poincaré dodecahedral space, Seifert–Weber). Walk through it in first person;
a map inset shows where you are in the Poincaré ball or on the stereographic sphere.

- Maths and validation: [docs/WRITEUP.md](docs/WRITEUP.md)
- Phase-0 study of Power Foam and the renderer architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Demo: [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) · progress log: [PROGRESS.md](PROGRESS.md)

## Run

```bash
# offline tools (Python 3.12): synthetic scene → export (with the curvature sweep)
python3.12 -m venv .venv && . .venv/bin/activate && pip install torch numpy scipy pyyaml pytest pillow
python tools/synth_scene.py --out scenes/synth_open --open
python tools/export_scene.py scenes/synth_open --out web/public/scenes/synth_open --curved --sweep 12 --kmax 0.042 --steiner-box 2.0 --steiner-iters 14 --centre-y 1.6
python -m pytest tests/python -q

# web app
cd web && npm install && npx vite            # http://127.0.0.1:5173  (?mode=explore|quiz|gallery skips the title screen)
npx vitest run                               # geometry / topology / gameplay tests
npx vite build                               # static site in web/dist
```

Validation page: `/harness.html` (GPU vs CPU reference PSNR, GLSL-vs-TypeScript geometry probe).

## Using a trained Power Foam checkpoint

Copy `output/<experiment>/` (needs `model.pt` and `config.yaml`) to `scenes/<name>/`, add a
`cameras.json` (see `scenes/synth_open/cameras.json`), then export as above. Everything else is
identical; the exporter derives frames, Steiner points and the curved adjacency itself.

## Controls

WASD move (shift run) · drag to look · **G** universe gallery (in Guess mode: the answer picker) · **M** enlarge the map (or click it) · click the **CRaFT** badge for the main menu ·
**`** developer panel · **Esc** help. Lagging? Lower **Render distance** or pick **fast** under Detail in the controls card.

## Layout

`tools/` Python exporters and reference renderers · `web/src/geometry` the κ-parameterised geometry
(mirrored by `web/shaders/geometry.glsl`) · `web/src/topology` fundamental domains and pairings ·
`web/src/render` passes · `web/src/game` camera, player (collision, compass maths) · `web/src/ui` map inset, gallery · `powerfoam/` the upstream repo as a submodule (unmodified).
