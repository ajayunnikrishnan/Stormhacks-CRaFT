# ARCHITECTURE.md — Phase 0 recon of Power Foam

Status: Phase 0 (read-only). Everything below was read from the paper
(`paper.pdf`, §3 + App. B–C) and the repo at `powerfoam/` (pinned as a git
submodule, commit `9639225`). No project code has been written yet.

Equation numbers "Eq. (n)" refer to the build prompt / `docs/WRITEUP.md`
numbering; "paper Eq. (n)" refers to the Power Foam paper.

---

## 0. TL;DR for the approval decision

1. **The repo is Python + PyTorch + NVIDIA Warp (CUDA only).** Nothing in it
   runs on this Mac or in a browser. Training, Steiner insertion, regular
   triangulation and the reference ray-traced images all need a Linux CUDA box.
   Our web renderer is a GLSL port of one ~130-line Warp kernel
   (`RayTracer.benchmark_kernel`) plus one ~60-line helper
   (`plane_intersection_fwd_local`) plus the Spherical Voronoi kernel.
2. **The ray tracer's view-dependent colour is not evaluated per ray.** It is
   evaluated once per frame, per *detail site*, from the direction
   `normalize(site − camera.eye)` (`SphericalVoronoi.forward`). The per-pixel
   kernel only blends precomputed per-site RGB. This is great news for WebGL:
   we mirror it with a small per-frame "SV pre-pass" into an N×k RGBA16F FBO,
   and the per-pixel walk never touches SV axes.
3. **Power distance is exactly `‖x − p‖² − r²`** (paper Eq. 2, `benchmark.py`
   `build_power_adjacency`, `rendering_math.ray_pface_intersect`). The
   build-prompt §3.4 curved rule `a_i = p_i / cs_κ(r_i)` has the correct flat
   limit against this definition. No re-parameterisation needed.
4. **Three things in the build prompt do not match the repo** (details §8):
   `render.py` is a mesh extractor, not an image renderer; the viewer only has
   a "Rasterize" mode and `RayTracer` has no `visualize()`; `benchmark.py
   --render_type raytrace` prints FPS but saves no images. We need a ~40-line
   script in `tools/` to dump reference ray-traced PNGs. Also the repo's
   Steiner routine differs from paper Alg. 1 in constants (10 iterations, not
   6; radius ×0.8; greedy de-overlap). We follow the **code**.
5. **Bytes per cell at paper settings (k=8 sites, 8 SV axes): 417 floats =
   1 668 B fp32 / ~846 B fp16.** A 500k-cell scene is ~420 MB in fp16 before
   adjacency. Proposed web budget: **150k cells, k=4, 4 SV axes, fp16 attrs,
   fp32 positions ≈ 36 MB + ~17 MB adjacency ≈ 53 MB raw**, est. 35–45 MB
   compressed (§6). Needs a retrain with `final_points: 150000`,
   `num_texel_sites: 4`, `sv_dof: 4`.
6. **Blockers only you can clear:** a CUDA machine + the trained scene
   (`{{SCENE_PATH}}` is unfilled), target GPU/OS, hours and team size.

---

## 1. Repo map (what matters to us)

| File | Role | Needed by us? |
|---|---|---|
| `powerfoam/raytrace.py` | `RayTracer.benchmark_kernel`: adjacency-walk ray tracer (Warp). **Reference implementation we port.** | Port to GLSL (Phase 1) |
| `powerfoam/rendering_math.py` | `ray_sphere_intersect`, `ray_pface_intersect(_diff)`, `ray_plane_intersect`. | Port |
| `powerfoam/rasterize.py` | 2 285 lines: tile rasterizer (train/test/viewer). Contains a duplicate `plane_intersection_fwd_local` identical to the ray tracer's. | Read-only; shading math is identical |
| `powerfoam/texture.py` | Older copy of the soft-Voronoi dipole shading with `tile_load` (max 8 sites). Not used by the kernels above. | No |
| `powerfoam/color_fn.py` | `SphericalVoronoi`: per-site directional RGB kernel. | Port (pre-pass shader) |
| `powerfoam/scene.py` | `PowerfoamScene`: parameters, activations (softplus β=100), quaternion→frame, `save_pt/load_pt`, densify/prune, Čech `rebuild_adjacency`. | Exporter reads it |
| `powerfoam/bvh.py` | Čech complex (all overlapping sphere pairs) via Warp BVH. Training/raster adjacency only. | No (we use the regular triangulation) |
| `powerfoam/camera.py` | `TorchCamera{eye,right,up,width,height}`, `get_ray_dir`. Pixel→ray convention we must replicate for PSNR. | Replicate |
| `benchmark.py` | `get_steiner_points`, `add_steiner_points`, `build_power_adjacency` (4D lift + qhull), FPS timing for raster/raytrace. | **Import** these functions from `tools/` |
| `render.py` | TSDF **mesh extraction** (Open3D). Not an image renderer. | No |
| `test.py`, `train.py` | Rasterizer-based eval/training. | Training on GPU box |
| `view.py`, `powerfoam/viewer.py` | ImGui/PyOpenGL viewer, rasterize mode only. | No |
| `configs/__init__.py` | `Params` dataclass (all flags). | Exporter reads `config.yaml` |

All GPU code is Warp kernels launched from PyTorch; there is no CPU fallback.

---

## 2. Checkpoint format

`output/<exp>/model.pt` = `torch.save` of a plain dict (`scene.py:save_pt`),
all tensors fp32 on CPU, N = number of cells, k = `num_texel_sites`,
D = `sv_dof`:

| key | shape | meaning (raw, pre-activation) |
|---|---|---|
| `points` | (N,3) | cell site / dipole face centre **p_i** |
| `radii` | (N,) | raw; **r_i = softplus(raw, β=100)** |
| `density` | (N,) | raw; **σ_i = softplus(raw, β=100)** |
| `quaternions` | (N,4) | (w,x,y,z), normalised on use; rotation columns give **n_i** (col 0), tangent **t_i** (col 1), bitangent **b_i** (col 2) |
| `texel_sites` | (N,k,2) | detail-site 2D coords **in units of r_i** in the (t_i,b_i) plane |
| `texel_sv_axis` | (N,k,3D) | per site, D axes; **‖axis‖ is the SV temperature**, direction is the SV site |
| `texel_sv_rgb` | (N,k,3D) | per site, D RGB values (centred at 0; +0.5 added at eval) |
| `texel_height` | (N,k) | displacement **in units of r_i** along n_i |
| `adjacency`, `adjacency_offsets` | CSR int32 | **Čech** graph (sphere overlaps) at save time — rasterizer only, not what the ray tracer uses |

Plus `output/<exp>/config.yaml` (frozen `Params`) and `points.ply`.
There is no stored normal, frame, Steiner point or regular triangulation; the
exporter derives all of them.

Derived quantities (exact formulas from `scene.py`):

```
r   = softplus(radii, 100)            σ = softplus(density, 100)
q   = quaternions / ‖quaternions‖ = (w,x,y,z)
n   = (1−2(y²+z²), 2(xy−zw), 2(xz+yw))            # normalised again
t   = (2(xy+zw), 1−2(x²+z²), 2(yz−xw))
b   = (2(xz−yw), 2(yz+xw), 1−2(x²+y²))
site_3d[i,j] = p_i + r_i·(texel_sites[i,j,0]·t_i + texel_sites[i,j,1]·b_i)
h[i,j]       = r_i · texel_height[i,j]
sv_temp[i,j,d] = ‖texel_sv_axis[i,j,d]‖ ;  sv_axis = texel_sv_axis / sv_temp
```

Training-time ordering: `sort_points()` Morton-sorts cells (GPU radix sort) for
cache locality. The exporter should re-sort on CPU (any Morton impl) since
locality matters for texture-cache hit rate in the shader.

---

## 3. Cell parameterisation (the exact model the shader must reproduce)

A cell *i* is: **(power cell of (p_i, r_i)) ∩ (ball of radius r_i about p_i) ∩
(dense half-space of the displaced dipole plane)**.

### 3.1 Power cell and faces

Power distance (paper Eq. 2, App. B Eq. 9): `pow(x, i) = ‖x − p_i‖² − r_i²`.
Bisector of i, j (`ray_pface_intersect`):

```
face_n      = p_j − p_i
face_offset = ½(‖p_j‖² − ‖p_i‖² + r_i² − r_j²)          # = pm_j − pm_i, pm = ½(‖p‖² − r²)
plane:  face_n · x = face_offset ;   ray: t = (face_offset − face_n·o) / (face_n·d),  dp = face_n·d
```

`benchmark.py` precomputes `adjacency_diff = (p_j − p_i, pm_j − pm_i)` **in
fp16** per directed edge. (Precision hazard for large scenes; we will compute
the difference in-shader from fp32 site data instead — same math, better
conditioned, and it makes the curvature slider cheap because only the per-site
texture changes.)

### 3.2 Sphere bound

`ray_sphere_intersect(o, d, p, r)` with unit d: standard quadratic. Returns
`hit=False` if no real roots or both roots < 0; clamps `t_near` to 0 if the
origin is inside. **Only the part of the power cell inside the sphere carries
density**; the walk still passes through the empty remainder.

### 3.3 Dipole + detail sites + displacement (paper §3.3–3.4, Eq. 3)

`plane_intersection_fwd_local(o, d, t_near, p, n, r, sites, rgbs, heights, k)`
(identical copies in `raytrace.py:43` and `rasterize.py:68`), `temp = 10`:

```
# pass 1: undisplaced plane  n·x = n·p
(_t_surf, _dp) = ray_plane_intersect(o, d, p, n, h=0)      # t = (n·(p−o) + h)/(n·d), dp = n·d
_t_query = t_near            if _dp >= 0
         = max(t_near,_t_surf) otherwise
x̄ = o + _t_query·d
w_j   = exp(−10 · ‖x̄ − site_3d_j‖² / r²)                    # 3D Euclidean distance, j = 1..k
h_out = Σ w_j h_j / max(Σ w_j, 1e−20)
# pass 2: displaced plane  n·x = n·p + h_out
(t_surf, dp) = ray_plane_intersect(o, d, p, n, h_out)
t_query = t_near if dp >= 0 else max(t_near, t_surf)
x = o + t_query·d
w_j   = exp(−10 · ‖x − site_3d_j‖² / r²)
rgb   = Σ w_j rgb_j / max(Σ w_j, 1e−20)                     # rgb_j = per-site SV colour (§4)
return (… h_out, t_surf, dp, rgb …)
```

Sign convention: the **dense half is `n·(x − p) ≤ h`** (behind the normal; the
normal loss, paper Eq. 15, pushes n toward the camera). The caller clips the
segment:

```
t_far  = min(t_surf, t_far)  if dp >= 0     # ray moving along +n: exits dense half at t_surf
t_near = max(t_surf, t_near) if dp <  0     # ray moving against n: enters dense half at t_surf
```

So the colour is sampled at the **entry point into the dense half**, once per
cell, and is constant along the segment. Note the soft-Voronoi weights use
`‖·‖²/r²` (Gaussian in units of the cell radius), whereas SV (§4) uses an
un-squared chord distance.

### 3.4 Volume integration along a segment

Piecewise-constant density: `density_integral` returns `−σ·(t_far − t_near)`.

```
dt = t_far − t_near
if hit and dt > 0:
    Δ = −σ·dt ;  α = 1 − exp(Δ)
    rgb_out += colour · α · T ;  log T += Δ
```

Termination: `T < transmittance_threshold` (1e-2 in `benchmark.py`, 1e-3 in
the viewer) or no exit face. Background colour added as `bkgd · T` (black in
all shipped configs). No depth/normal output in the ray tracer; the
rasterizer's visualisation kernel also accumulates `normal·α·T` and a
transmittance-quantile depth (`t_near + log(T/q)/σ`), which we can copy for our
normals and collision probe.

---

## 4. Spherical Voronoi colour (paper §3.4 Eq. 4 + Di Sario et al.)

`color_fn.py:spherical_voronoi_fwd_kernel`, launched **once per frame over all
N·k detail sites**, not per pixel:

```
dir = normalize(site_3d − camera.eye)
if dot(dir, camera_forward) < fov_cos_cutoff: rgb = (0.5,0.5,0.5)   # frustum cull only
w_d = exp(−temp_d · ‖dir − axis_d‖)        # chord distance, NOT squared, d = 1..D
rgb = max(Σ w_d rgb_d / Σ w_d + 0.5, 0)    # per channel clamp at 0
```

Consequences for us:

* The per-pixel walk needs only `rgb[i,j]` (k RGB values per cell), so SV axes
  never enter the hot loop.
* In curved space with the camera at the origin, the "direction to the site"
  is the **transported tangent of the geodesic from o to site_3d at the site**,
  Eq. (4), expressed in the site's local frame via `M_p⁻¹` (Eq. 3.6). That is
  one closed-form evaluation per site per frame — same cost structure as the
  repo.
* With tiling, the same cell is seen along several geodesics (different
  copies). The pre-pass gives the direct copy's direction only. Plan: pre-pass
  is the default (Low/Medium presets); High preset evaluates SV per pixel for
  the shaded cell (D·k extra fetches) — document as a quality knob.
* We do **not** apply the FOV cull (cells can be seen through walls from any
  direction).

---

## 5. The ray-trace walk (`RayTracer.benchmark_kernel`) — order of operations

Inputs: `start_point_idx` (argmin over cells of `‖p − eye‖² − r²`, i.e. exact
brute-force point location of the camera), fp16 `adjacency_diff`, CSR
adjacency from the **regular triangulation + Steiner points** (not Čech).

```
ray_d = normalize(x·right + y·up + forward),   x = 2j/(W−1) − 1,  y = 1 − 2i/(H−1)
prim = start ; log_T = 0 ; pt_near = 0
loop:
    T = exp(log_T) ; if T < thr or prim == INT_MAX: break
    (p, r) = spheres[prim]
    hit, t_near, t_far = ray_sphere_intersect(eye, d, p, r)
    if ‖p − eye‖ < 4r: hit = False                       # (!) near-camera cull, ray tracer only
    next = INT_MAX ; pt_far = 1e10
    for each neighbour j of prim:                        # CSR, n_adj typically 20–40
        (t_face, dp) = ray_pface_intersect_diff(eye, d, diff_ij, pm_diff_ij)
        if dp >= 0 and t_face < pt_far: next = j ; pt_far = t_face     # first exit face
        t_far  = min(t_face, t_far)  if dp >= 0
        t_near = max(t_face, t_near) if dp <  0          # clip sphere interval to the power cell
    (n, σ) = nsigma[prim]
    if not hit or t_near > t_far or σ < 1e-3:            # empty cell or no overlap: just advance
        prim = next ; pt_near = max(pt_near, pt_far) ; continue
    (h, t_surf, dp, colour) = plane_intersection_fwd_local(...)   # §3.3
    clip [t_near,t_far] by the displaced dipole plane (§3.3)
    prim = next ; pt_near = max(pt_near, pt_far)
    if hit and t_far − t_near > 0: accumulate (§3.4)
rgb += bkgd · exp(log_T)
```

Observations the GLSL port must preserve or consciously deviate from:

* The check `t_face > pt_near` is commented out. Correct for exact arithmetic
  on a convex cell (any dp>0 face lies ahead), but fp16 `pm_diff` can violate
  it; with fp32 in-shader differences we are safer. In S³ the exit solve
  (Eq. 6, `tn_κ` periodic in π) must pick the first root ahead of the current
  arc length explicitly.
* "No exit face" (`next == INT_MAX`) means the ray left the convex hull of
  all sites and terminates with background. In our tiled spaces the scene is
  centred inside the fundamental domain, so this should not happen before a
  domain face; we handle it by jumping straight to the domain-face
  intersection.
* The 4r near-camera cull is a visual hack (skips shading of cells whose
  centre is within four radii of the eye). Rasterizer doesn't have it. We'll
  expose it as a flag, default on in flat-reference mode for PSNR parity, off
  (or replaced by a proper geodesic-distance rule) in the game.
* Empty cells (σ < 1e-3 after softplus) skip all shading work — Steiner cells
  are saved with raw density −10 (σ ≈ 0), so they cost one sphere test plus
  the neighbour loop only.
* Threads are 8×8 tiles for memory coherence; a fragment shader gets that for
  free.

---

## 6. Adjacency, empty space, Steiner points

### 6.1 Two different graphs

| Use | Graph | Built by | Why |
|---|---|---|---|
| Training & rasterization | **Čech complex** (all pairs with `‖p_i − p_j‖ < r_i + r_j`) | `bvh.py` (Warp BVH), rebuilt every 1–20 iterations | Superset of the α-complex; cheap; extra faces never cut the bounded cell (paper Fig. 5–6) |
| Ray tracing | **Regular (weighted Delaunay) triangulation** of all sites incl. Steiner points | `benchmark.py:build_power_adjacency`: lift to `(p, ‖p‖² − r²)`, scipy `ConvexHull` (qhull) in 4D, keep facets with normal `w < 0` (lower hull), take tetra edges | Rays traverse faces outside the sphere bounds (paper App. C) |

`alpha_complex=True` additionally filters edges to overlapping spheres (used
for the rasterizer benchmark). The Čech graph in `model.pt` is **not** usable
for ray tracing.

### 6.2 Steiner points — the code (follow this, not paper Alg. 1 literally)

`benchmark.py:get_steiner_points(points, radii, cameras)`:

```
for 10 iterations:                                      # paper says 6
    all = scene ∪ steiner so far ; KD-tree on all
    sample 0.25·N points ~ Normal(mean(points), 0.5·std(points))   # paper: same mean/std
    k=32 nearest; r̂ = min_j (‖ŝ − p_j‖ − r_j) ; nearest = argmin ; ratio = r̂ / r_nearest
    keep 2 < ratio < 6
    greedy farthest-point filter so pairwise overlap (r_a + r_b − d)/(r_a + r_b) ≤ 0.1
    append kept with radius 0.8·r̂
raw radius = inverse_softplus(0.8 r̂, β=100) ; raw density = −10 ; zero texels/rgb/height
```

Note `r̂` here is distance-to-sphere-surface (`‖ŝ − p‖ − r`), not the power
distance `‖ŝ − p‖² − r²` written in Alg. 1. Paper reports 53 → 37 average
ray–cell intersections and 113 → 185 FPS on Bonsai. These functions are pure
PyTorch/scipy on whatever device the tensors live on; **they run on CPU torch
on this Mac**, so the exporter does not need CUDA once a checkpoint exists.

### 6.3 What changes for curved space (ties to build-prompt §3.5)

* Steiner insertion stays flat (done in the scene's own Euclidean coordinates
  before embedding).
* `build_power_adjacency` is exactly the "Euclidean regular triangulation of
  weighted sites" we need; we feed it the **projective-model transformed
  sites/weights** for each κ sample instead of raw ones, and union the edge
  sets. For S³ we instead take the 4D convex hull of `{a_i}` directly (also
  qhull). No CGAL needed.

---

## 7. Flat-space assumptions in the reference implementation (complete list)

Each one needs a curved replacement or a documented approximation:

1. Rays are `o + t·d`; `t` is Euclidean length. → Eq. (3), arc length.
2. Sphere bound via quadratic in `t`. → Eq. (7).
3. Power bisector is the radical plane `(p_j − p_i)·x = pm_j − pm_i`. → linear
   hyperplane `⟨x, a_i − a_j⟩_κ = 0` with `a = p / cs_κ(r)`; Eq. (6).
4. Power distance `‖x−p‖² − r²` for start-cell location. → `κ⟨x, a_i⟩` rule
   (§3.4 of prompt); flat-limit test required.
5. Dipole plane `n·x = n·p + h` with Euclidean offset `h` along `n`. → the
   undisplaced plane through p with normal n is a linear hyperplane
   `⟨x, N⟩_κ = 0` (N = n transported to p) → Eq. (6) exact; displacement `h`
   handled in the local frame (prompt §3.11).
6. Detail-site positions `p + r(u·t + v·b)` in the Euclidean tangent frame. →
   exp map at p (Eq. 3.6), exact per site.
7. Soft-Voronoi weights use Euclidean `‖x − site‖²/r²` in ℝ³. → log map to
   the tangent space at p (local-flat approximation, error ~ |κ|·r²).
8. SV direction `normalize(site − eye)` and chord distance on S². → transported
   geodesic tangent at the site, in the local frame (exact).
9. Camera: `x·right + y·up + forward` in world ℝ³; eye anywhere. → camera at
   origin, world moved by isometry W (prompt §3.7).
10. Near-camera cull `‖p − eye‖ < 4r`. → geodesic distance or drop.
11. Transmittance `exp(−σ·Δt)`, Δt Euclidean. → arc length; unchanged form.
12. Steiner sampling Gaussian in ℝ³, Euclidean KD-tree. → keep flat (before
    embedding).
13. Regular triangulation via 4D lift. → projective-model transform (H³) /
    4D hull (S³), union over κ.
14. fp16 edge differences in absolute world units. → compute in-shader, fp32.
15. Morton sort, tile layout, depth quantile in `t`. → cosmetic / unchanged.
16. Pixel grid maps image edges to ±1 (`2j/(W−1) − 1`), i.e. pixel *centres*
    at the extreme values. Must be replicated exactly for the Phase 1 PSNR
    test; the game can use the conventional half-pixel offset.

---

## 8. Discrepancies between the build prompt and the repo

| Prompt says | Repo reality | What we do |
|---|---|---|
| `render.py` renders reference images | `render.py` = TSDF mesh extraction | Reference images via our `tools/render_reference.py` calling `model.raytracer.benchmark(...)` (same inputs `benchmark.py` builds) and saving PNG + camera JSON |
| viewer can show the ray tracer | `RENDER_MODES = ["Rasterize"]`; `RayTracer` has no `visualize()`, so `forward_visualization(render_mode="raytrace")` would raise `AttributeError` | Don't rely on the viewer |
| Paper Alg. 1: 6 iterations, power-distance radius | Code: 10 iterations, surface-distance radius, 0.8 shrink, greedy de-overlap | Follow the code; cite both in the writeup |
| "Verify Power Foam's power-distance definition" | `‖x − p‖² − r²` exactly | Prompt §3.4 flat-limit check holds as stated; unit test still required |
| Prompt §3.11: SV lookup per hit using transported direction | Repo evaluates SV **per detail site per frame** from the eye | Mirror the repo (pre-pass); optional per-pixel mode |
| Scene has `normals` | Only quaternions; frame derived | Exporter derives n, t, b |
| `tools/` must build adjacency with CGAL | Repo already does it with scipy qhull in 4D | Reuse `build_power_adjacency` |

One repo nit, not ours to fix: `texture.py` hard-codes `max_sites = 8` with
`tile_load`, but neither live kernel uses it; `rasterize.py`/`raytrace.py`
loop over `num_texel_sites` directly, so **k = 4 trains and renders fine**
(paper Table 3 confirms the ablation exists).

---

## 9. Bytes per cell and the web scene budget

Per cell, floats (k sites, D SV axes):

| field | floats | k=8,D=8 | k=4,D=4 |
|---|---|---|---|
| p (fp32 always) | 3 | 3 | 3 |
| r, σ | 2 | 2 | 2 |
| frame (n or q) | 3–4 | 4 | 3 (n only; t,b folded into site offsets) |
| site offsets (3D, relative to p) | 3k | 24 | 12 |
| heights | k | 8 | 4 |
| SV axes (raw, norm = temp) | 3kD | 192 | 48 |
| SV rgb | 3kD | 192 | 48 |
| **total floats** | | **425** | **120** |
| bytes, fp16 attrs + fp32 p | | 856 B | **246 B** |
| adjacency (regular triangulation, avg deg ≈ 28 × int32) | | ~112 B | ~112 B |

(Repo ships `texel_sites` as 2D (2k floats); we export 3D offsets (3k) to
avoid per-pixel frame maths. k=8,D=8 is 417 floats in the repo's own layout.)

Totals:

| scene | cells | k, D | attrs | adjacency | raw total | est. brotli |
|---|---|---|---|---|---|---|
| paper indoor | 500k | 8, 8 | 428 MB | ~56 MB | ~480 MB | way over |
| **proposed default** | **150k** | **4, 4** | **37 MB** | **17 MB** | **~54 MB** | **35–45 MB** |
| fallback (quality) | 150k | 4, 8 | 66 MB | 17 MB | ~83 MB | 60–70 MB |
| fallback (size) | 100k | 4, 4 | 25 MB | 11 MB | ~36 MB | 25–30 MB |

fp16 is validated by the repo itself: `benchmark.py` runs with
`attr_dtype="half"`, i.e. the reported ray-tracing numbers already use fp16
attributes (and fp16 absolute texel-site positions, which we improve on).

Risks: paper Table 3 ablates k (4 vs 8 = −0.4 dB) but not D; D=4 is a guess.
Mitigation: train two checkpoints (D=4, D=8) on the GPU box; the exporter
handles either, the shader pre-pass takes D as a uniform.

Texture plan (all 2D with row wrapping, `MAX_TEXTURE_SIZE` queried at startup;
16384² = 268M texels, so 150k×4×4 = 2.4M texels per SV texture is fine):

| texture | format | size | read in |
|---|---|---|---|
| `site4` (a_i / p_i in ℝ⁴ for current κ, + r) | RGBA32F ×2 | N | walk (self + neighbour) |
| `nsigma` (transported n, σ) | RGBA16F | N | walk |
| `siteoff` (3D offsets ×k) + `height` | RGBA16F | N·k | walk (shaded cells only) |
| `adj_offsets`, `adj_index` | R32UI | N+1, E | walk |
| `sv_axis`, `sv_rgb` | RGB16F | N·k·D | SV pre-pass only |
| `site_rgb` (pre-pass output) | RGBA16F FBO | N·k | walk (shaded cells only) |

Per-κ CPU work when the slider moves: recompute `site4` for N sites (exp map)
and re-upload one RGBA32F texture (2.4 MB) — cheap. Adjacency is the union
graph and never changes.

---

## 10. Stack confirmation and WebGL2 feasibility

Confirmed as in prompt §2: C++17 + GL ES 3.0 / GLSL ES 3.00, Emscripten →
WebGL2, native GLFW build for debugging, Dear ImGui, Python tools.

WebGL2 specifics that the port needs, and why they are fine:

* `texelFetch` on float/uint textures: core in WebGL2 (`EXT_color_buffer_float`
  needed only to *render into* RGBA16F/32F FBOs — the SV pre-pass needs it;
  universally available on desktop Chrome).
* Dynamic loops are legal in GLSL ES 3.00, but ANGLE/D3D can be slow or
  reject huge unrolls; we use `for (int i = 0; i < MAX_STEPS; ++i)` with
  `MAX_STEPS` a compile-time constant (e.g. 256 cells, 64 neighbours) and
  `break`.
* No scatter needed anywhere: the ray tracer is a pure gather. The SV
  pre-pass is a gather too (one fragment per detail site).
* Expected cost: paper reports ~37 cell visits per ray with Steiner points,
  each visit ≈ 1 sphere test + ~28 plane tests + (shaded only) 2×k Gaussian
  weights. Roughly 1.2k FLOPs and ~35 dependent texel fetches per pixel; at
  1080p ≈ 2 Gpix-fetches/frame. Mid-range desktop GPUs sustain this at
  30–60 FPS in WebGL if cells are Morton-sorted; dynamic resolution scaling
  (50–100%) is the safety valve. We measure in Phase 1 before committing.

---

## 11. Proposed exporter (`tools/export_scene.py`, Phase 1)

Runs on CPU torch (no CUDA, no Warp import):

1. Load `config.yaml` + `model.pt`; apply activations; derive n, t, b.
2. `get_steiner_points` (from `benchmark.py`, CPU tensors) → append cells
   with σ = 0, uniform grey rgb, zero heights.
3. Morton-sort all cells.
4. Flat: `build_power_adjacency(points, r)` (regular triangulation).
   Curved (Phase 3): transform to projective model per κ sample, union edges;
   S³ via 4D hull of `{a_i}`.
5. Verify adjacency vs brute-force nearest-power-site on 1e5 samples.
6. Write `scene.bin` (little-endian, header + sections, fp16 attrs) +
   `scene.json` (counts, k, D, bbox, suggested floor height) and gzip/brotli.

Reference renders (`tools/render_reference.py`, GPU box): rebuild exactly the
inputs `benchmark.py:test()` builds, call `model.raytracer.benchmark(...)` for
a list of cameras, save PNG + camera JSON, and compute PSNR against our
renderer's output of the same cameras (target > 40 dB).

---

## 12. Open questions / blockers (need answers before Phase 1)

1. **GPU machine**: where do training, Steiner/triangulation (CPU ok) and the
   reference ray-traced renders run? This Mac cannot run any Warp kernel.
2. **Scene**: path to an existing checkpoint (any scene) to start Phase 1
   now, and permission to retrain at the §9 budget.
3. **Target GPU/OS** for the 30 FPS target, hours, team size.
4. **D = 4 vs 8**: train both if GPU time allows.


---

## 13. As built (added after Phase 9)

The plan above was executed with one change: the app is **TypeScript + WebGL2 (Vite)** instead
of C++/Emscripten — same GLSL ES 3.00 shaders, same data-texture layout, no toolchain risk. The
exporter follows §11 exactly (Steiner points, Morton sort, regular triangulation, fp16 attributes)
and additionally ships the union adjacency over a curvature sweep (§3.5 of the writeup) and a
box-sampled Steiner option for open scenes. Frame pipeline as built:

```
sv_prepass_curved (N·k texels)  →  walk_curved (MRT G-buffer: baked+T, hit, normal, meta)
                                →  shadow (½ res, ≤4 lamps)  →  shade (Eq. 8, fog, tone map)
```

Per-κ variants of every curved shader are compiled with `#define KAPPA` so the branches fold.
See docs/WRITEUP.md §§8, 11–14 for the numerical findings, validation and performance tables.
