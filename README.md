# CRaFT: Curved Radiance Foam Tracing

*Exact Geodesic Ray Tracing of Radiance Foams in Constant-Curvature Spaces*

Hackathon entry for Huawei Custom Challenge #1, **Beyond Euclid**. A photographic scene reconstructed
with [Power Foam](https://github.com/theialab/powerfoam) is ray traced in the browser (WebGL2) along
**exact geodesics** of hyperbolic (H³), flat (E³) and spherical (S³) space. A slider bends the
curvature continuously, a gallery drops you into closed universes (3-torus, half-turn space, Klein
space, {4,3,4}, {4,3,5}, the tesseract tiling of S³, Poincaré dodecahedral space, Seifert–Weber
space), and a quiz asks you to identify the universe you were dropped into by walking around.

Two scenes ship: a synthetic test room and **Treehill** from the Mip-NeRF 360 dataset, trained with
Power Foam (300k cells) and validated against Power Foam's own CUDA ray tracer at 38–44 dB PSNR.

- Maths and validation in full: [docs/WRITEUP.md](docs/WRITEUP.md)
- Power Foam study and renderer architecture: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Training a scene on a rented GPU: [docs/TRAINING_VASTAI.md](docs/TRAINING_VASTAI.md)
- Demo script: [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) · progress log: [PROGRESS.md](PROGRESS.md)

## The idea in one paragraph

Power Foam represents a scene as a **power diagram**: a cell per site (pᵢ, rᵢ), and a ray is rendered
by walking from cell to cell through the planar faces, accumulating the cell's colour and density
(Govindarajan et al., 2026). The key observation behind CRaFT is that this walk survives a change of
geometry. Embed H³ and S³ in ℝ⁴ with the form ⟨x,y⟩_κ = κx₀y₀ + x̄·ȳ and define aᵢ = pᵢ / cs_κ(rᵢ)
(cs = cos on S³, cosh on H³). Then

    cell(x) = argmax_i ⟨x, aᵢ⟩_κ

reproduces the curved power distance (H³: minimise cosh d / cosh r; S³: maximise cos d / cos r; its
flat limit is exactly ‖x−p‖² − r²), and the bisector of two cells, ⟨x, aⱼ − aᵢ⟩_κ = 0, is a **linear
hyperplane through the origin of ℝ⁴**. So the cell faces are still planes in the model, the adjacency
is still a regular triangulation (a 4-D convex hull on S³; a weighted Delaunay in the Klein model on
H³), and the ray walker is *the same algorithm* as the flat one, with the straight ray replaced by
the closed-form geodesic. Nothing is approximated along the ray; curvature enters only through two
one-line intersection formulas.

## The maths that runs on the GPU

Equation numbers match the code comments and [docs/WRITEUP.md](docs/WRITEUP.md).

**Model spaces (Eq. 1–2).** One geometry layer serves all three spaces: ambient ℝ⁴, the bilinear form
⟨x,y⟩_κ = κx₀y₀ + x₁y₁ + x₂y₂ + x₃y₃, H³ the hyperboloid ⟨x,x⟩ = −1, S³ the unit sphere, E³ the
affine chart (1, x̄). Generalised trigonometry sn_κ, cs_κ (sinh/cosh, sin/cos, t/1); distance from
cs_κ d = κ⟨x,y⟩_κ, evaluated through the chord ⟨x−y, x−y⟩_κ so tiny distances stay accurate. The
curvature slider sets k in 1/m²; a scene of size L is embedded at size √|k|·L, so k → 0 tends to the
flat picture from both sides.

**Geodesics (Eq. 3–5).** From o′ with unit tangent v: γ(t) = cs_κ(t)·o′ + sn_κ(t)·v, and the
parallel-transported direction γ′(t) = −κ sn_κ(t)·o′ + cs_κ(t)·v is what every directional lookup
uses (view-dependent colour, the compass). The unit tangent from x toward y is
u = (y − cs_κ(d)·x) / sn_κ(d).

**Closed-form intersections (Eq. 6–7).** For a plane {w·x = 0} with A = w·o′, B = w·v the crossing
solves A cs_κ(t) + B sn_κ(t) = 0, i.e. tn_κ(t) = −A/B: in H³ t = ½ ln((B−A)/(A+B)) (exists iff
|B| > |A|), in S³ exits at φ − π/2 + 2πk with φ = atan2(B, A), so the walker explicitly asks for
"the first exit after t" and "the last entry before t" on the periodic sphere. The ball test
d(γ(t), p) ≤ r becomes A cs_κ(t) + B sn_κ(t) ≥ κ cs_κ(r): a quadratic in eᵗ on H³, an arc
[φ − α, φ + α] with α = acos(cos r / √(A²+B²)) on S³.

**Flat-trained foam in a curved world (§5).** Cell *traversal* is exact in the curved space; cell
*shading* (detail sites, dipole density, spherical-Voronoi colour) is evaluated in each cell's own
flat frame reached by the isometry that carries the origin to the site, with the ray's transported
direction expressed in that frame. The error is local, bounded by ≈ 2.8·κR² for a cell of radius R.

**Precision in fp32 (§8).** Near k = 0 the curved quantities are O(s²) differences of O(1) numbers,
and the first implementation rendered garbage below s ≈ 10⁻³. Storing a₀ − 1 instead of a₀ and
rewriting Eq. 7 around the chord A′ = −½⟨o′−p, o′−p⟩_κ (same equations, different order of
operations) brought GPU fp32 to within 45–66 dB of an fp64 CPU reference across the whole slider,
and made the switch to the E³ code path invisible (43–52 dB at k = ±10⁻⁶).

**Closed universes (§7).** Each universe is a fundamental polyhedron with face pairings: when the
ray (or the walker) leaves through face f it is re-expressed by the isometry g_f onto the paired
face. The 3-torus, half-turn and Klein spaces are boxes in E³; {4,3,5} is the right-angled
hyperbolic cube (inradius 0.5306), the tesseract tiling of S³ a cube of inradius π/4, Poincaré
dodecahedral space a spherical dodecahedron of inradius π/10 glued with a 36° twist, Seifert–Weber a
hyperbolic dodecahedron (inradius 0.996) glued with a 108° twist. Dihedral angles, pairings and
edge cycles are tested to 10⁻⁶°. The compass carried by the walker measures holonomy = κ·Area, a
Gauss–Bonnet check that doubles as a game mechanic.

**Validation.** Flat GPU walk vs a numpy port of the Power Foam kernel 72–74 dB; curved walker at
κ = 0 vs flat 81–95 dB; GLSL vs TypeScript geometry 0 mismatches on 27 function×κ combinations;
closed-form roots vs bisection 10⁻⁷; and, on the real Treehill checkpoint, the browser renderer vs
**Power Foam's own CUDA ray tracer** 43.9 / 41.3 / 44.4 / 43.4 dB (150k-cell model) and
38.4–39.9 dB (300k-cell model) at 1267×832.

## Run

```bash
# offline tools (Python 3.12): synthetic scene → export (with the curvature sweep)
python3.12 -m venv .venv && . .venv/bin/activate && pip install torch numpy scipy pyyaml pytest pillow matplotlib
python tools/synth_scene.py --out scenes/synth_open --open
python tools/export_scene.py scenes/synth_open --out web/public/scenes/synth_open --curved --sweep 12 --kmax 0.042 --steiner-box 2.0 --steiner-iters 14 --centre-y 1.6
python -m pytest tests/python -q

# web app
cd web && npm install && npx vite            # http://127.0.0.1:5173  (?scene=synthetic|treehill&mode=explore|quiz|gallery)
npx vitest run                               # geometry / topology / gameplay tests
npx vite build                               # static site in web/dist
```

Validation page: `/harness.html?scene=scenes/treehill/scene.json&refs=scenes/treehill/refs` compares
the GPU walk against reference renders camera by camera.

## Using a trained Power Foam checkpoint

Train with the upstream repo (see [docs/TRAINING_VASTAI.md](docs/TRAINING_VASTAI.md) for a 45-minute
recipe on a rented RTX 4090), render references and dump the cameras with
`tools/render_reference.py`, then

```bash
python tools/canonicalize_scene.py scenes/<exp>_raw --out scenes/<name> --cameras scenes/<exp>_raw/refs_train/cameras_all.json --cam-height 1.5 --crop-radius 12 --crop-below 1.5 --crop-above 12
python tools/export_scene.py scenes/<name> --out web/public/scenes/<name> --curved --sweep 12 --kmax 0.015 --steiner-box 2.0 --steiner-iters 14 --centre-y 1.6 --walk-box -4 0 -4 4 3.2 4 --background 0.72 0.74 0.78 --render-distance 30
```

The first step applies one similarity transform to the whole checkpoint (points, radii through the
softplus, densities, dipole frames, view-dependent axes, cameras) so COLMAP's arbitrary units become
metres with y up and the floor at 0; the floor plane and scale are measured from the training cameras
and the floor cells. Register the scene in the `SCENES` list in `web/src/main.ts`.

## Controls

WASD move (shift run) · drag to look · **G** universe gallery (in Guess mode: the answer picker) ·
**M** enlarge the map (or click it) · **C** collapse the controls · click the **CRaFT** badge for the
main menu · **`** developer panel · **Esc** help. Lagging? Lower **Render distance** or pick **fast**
under Detail.

## Layout

`tools/` Python exporters, canonicalisation and reference renderers · `web/src/geometry` the
κ-parameterised geometry (mirrored line for line by `web/shaders/geometry.glsl`) · `web/src/topology`
fundamental domains and pairings · `web/src/render` passes (view-dependent colour pre-pass, geodesic
walk, shading) · `web/src/game` camera and player · `web/src/ui` map inset and gallery ·
`powerfoam/` the upstream repository as an unmodified submodule.

## References

- S. Govindarajan, D. Rebain, D. Verbin, K. M. Yi, A. Prabhu, A. Tagliasacchi. **Power Foam:
  Unifying Real-Time Differentiable Ray Tracing and Rasterization.** arXiv:2604.24994, 2026.
  Code: https://github.com/theialab/powerfoam. CRaFT ports the exporter-side maths (activations,
  Steiner points, regular-triangulation adjacency) and the ray-walk kernel faithfully; the curved
  cell rule, geodesic intersections and topology layer are ours.
- J. T. Barron, B. Mildenhall, D. Verbin, P. P. Srinivasan, P. Hedman. **Mip-NeRF 360: Unbounded
  Anti-Aliased Neural Radiance Fields.** CVPR 2022. The Treehill capture is from this dataset.
- F. Nielsen, R. Nock. Hyperbolic Voronoi diagrams made easy. ICCSA 2010 (the Klein-model
  reduction used to reuse the Euclidean regular-triangulation builder for H³).
- W. P. Thurston, *Three-Dimensional Geometry and Topology*; J. R. Weeks, *The Shape of Space*
  (the fundamental domains and their gluings).

```bibtex
@article{govindarajan2026powerfoam,
  title   = {Power Foam: Unifying Real-Time Differentiable Ray Tracing and Rasterization},
  author  = {Govindarajan, Shrisudhan and Rebain, Daniel and Verbin, Dor and Yi, Kwang Moo and Prabhu, Anish and Tagliasacchi, Andrea},
  journal = {arXiv},
  year    = {2026}
}
```
