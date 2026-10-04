# CRaFT: Curved Radiance Foam Tracing

*Exact geodesic ray tracing of radiance foams in constant-curvature spaces, and a game that asks you which universe you are standing in.*

**Live:** https://ajayunnikrishnan.github.io/Stormhacks-CRaFT/ · **Code:** https://github.com/ajayunnikrishnan/Stormhacks-CRaFT

## Inspiration

Every "non-Euclidean" demo we could find renders synthetic geometry: coloured cubes in hyperbolic space, a corridor that loops back on itself. We wanted the opposite: take a **real place**, captured with a phone camera, and let you walk around in it while the laws of geometry change under your feet. Bend space until the park bench curves away from you, step through a wall and come back from the opposite side, and then ask the hard question: *from what you can see, which universe is this?*

Power Foam (Govindarajan et al., 2026) made that possible. It reconstructs a photographed scene as a **radiance foam**: a power diagram whose cells carry colour and density, rendered by walking a ray from cell to cell. The crucial thing about that representation is that nothing in it is a triangle or a splat that has to be projected. A ray simply visits cells in order. That is a representation you can take to a different geometry, if you can show that its cells survive the trip.

## What it is

CRaFT is a WebGL2 app with three modes on top of one renderer.

- **Explore.** First-person walking (WASD, mouse look, collisions, floor following) through a Power Foam scene, with a slider that sets the curvature $k$ of space in $1/\mathrm{m}^2$, continuously from hyperbolic through flat to spherical. Every ray is traced along an exact geodesic of $\mathbb{H}^3$, $\mathbb{E}^3$ or $\mathbb{S}^3$; nothing is approximated along the ray.
- **Choose a universe.** A gallery of closed 3-manifolds built as fundamental polyhedra with face pairings: the 3-torus, the half-turn space, the Klein space, the $\{4,3,4\}$ cubic tiling, the right-angled hyperbolic cube $\{4,3,5\}$, the tesseract tiling of $\mathbb{S}^3$, the Poincaré dodecahedral space (36° twist) and the Seifert–Weber space (108° twist). The gallery pictures are rendered from the same face data the ray walker uses, and the wall tints in-world match the colours on the polyhedron.
- **Guess the universe.** You are dropped into one of them at random, with the tints off and the map hiding the domain. You walk, watch how the room repeats, how many copies meet at an edge, whether you come back mirrored or rotated, and then pick the polyhedron you think you are in. The reveal shows the answer and keeps score. It is a physically motivated version of the cosmologist's problem: the topology of space is not visible locally, only through the pattern of its repetitions.

Two scenes ship: a synthetic test room, and **Treehill** from the Mip-NeRF 360 dataset, trained with Power Foam on a rented RTX 4090 and validated in our browser renderer against Power Foam's own CUDA ray tracer at 38 to 44 dB PSNR.

## The mathematical trick

Power Foam's cell of a site $(p_i, r_i)$ is the set of points where the power distance $\|x-p_i\|^2 - r_i^2$ is minimal. In flat space the bisector of two cells is a plane, so a ray leaves a cell through a planar face and the walker only ever solves ray–plane intersections.

The question is whether anything like that survives in curved space. Embed $\mathbb{H}^3$ and $\mathbb{S}^3$ in $\mathbb{R}^4$ with the bilinear form

$$\langle x, y\rangle_\kappa = \kappa\, x_0 y_0 + x_1 y_1 + x_2 y_2 + x_3 y_3,$$

so that $\mathbb{S}^3$ is $\langle x,x\rangle = 1$ with $\kappa=+1$ and $\mathbb{H}^3$ is the hyperboloid $\langle x,x\rangle = -1$ with $\kappa=-1$. Write $\mathrm{cs}_\kappa$ for $\cos$ on the sphere and $\cosh$ on the hyperboloid, and define a lifted site

$$a_i = \frac{p_i}{\mathrm{cs}_\kappa(r_i)} .$$

Then

$$\mathrm{cell}(x) = \arg\max_i\ \langle x, a_i\rangle_\kappa$$

is the curved power diagram: on $\mathbb{H}^3$ it minimises $\cosh d / \cosh r$, on $\mathbb{S}^3$ it maximises $\cos d / \cos r$, and both expand to $1 \mp \tfrac{1}{2}(d^2 - r^2) + O(s^4)$, which is exactly the Euclidean power distance in the flat limit (we test that expansion numerically). The payoff is one line:

$$\text{bisector}(i,j) = \{\, x : \langle x,\ a_j - a_i\rangle_\kappa = 0 \,\},$$

a **linear hyperplane through the origin of $\mathbb{R}^4$**. Cell faces are therefore still planes in the model. The adjacency is still a regular triangulation (the normal fan of $\mathrm{conv}\{a_i\}$, i.e. a 4-D convex hull, on the sphere; a weighted Delaunay triangulation in the Klein model on the hyperboloid). And the ray walker is **the same algorithm as Power Foam's**, with the straight ray replaced by the geodesic

$$\gamma(t) = \mathrm{cs}_\kappa(t)\, o + \mathrm{sn}_\kappa(t)\, v, \qquad \gamma'(t) = -\kappa\, \mathrm{sn}_\kappa(t)\, o + \mathrm{cs}_\kappa(t)\, v .$$

Crossing a face $w\cdot x = 0$ with $A = w\cdot o$ and $B = w\cdot v$ means solving $A\,\mathrm{cs}_\kappa(t) + B\,\mathrm{sn}_\kappa(t) = 0$, i.e. $\tan_\kappa t = -A/B$, which is $t = \tfrac12 \ln\frac{B-A}{A+B}$ on $\mathbb{H}^3$ and a periodic family on $\mathbb{S}^3$ (the walker asks for "the first exit after $t$" explicitly, because on a sphere a ray leaves and re-enters every half-space forever). The ball test $d(\gamma(t), p) \le r$ becomes a quadratic in $e^t$ or an arc of angle $\arccos(\cos r/\sqrt{A^2+B^2})$. Those two formulas are the entire difference between the flat and the curved renderer.

A scene trained in flat space is embedded at scale $s = \sqrt{|k|}$, so cell *traversal* is exact in the curved space while cell *shading* (the detail sites, the dipole density, the view-dependent colour) is evaluated in each cell's own flat frame, reached by the isometry that carries the origin to the site, with the ray's parallel-transported direction $\gamma'$ expressed in that frame. The error is local and bounded by about $2.8\,\kappa R^2$ for a cell of radius $R$.

The closed universes add one more operation: when a ray or the walker leaves the fundamental polyhedron through face $f$, it is re-expressed by the isometry $g_f$ onto the paired face. Dihedral angles, pairings and edge cycles are tested to $10^{-6}$ degrees. The compass the walker carries is parallel-transported, so after a loop it has turned by $\kappa\cdot\mathrm{Area}$, which is Gauss–Bonnet doubling as a sanity check.

## How we built it

- A single geometry layer, written twice: `space.ts` in float64 and `geometry.glsl` in float32, line for line, with the same equation numbers. A probe shader evaluates every GLSL function on 4096 random inputs per curvature and compares with TypeScript (0 mismatches).
- An offline exporter in Python that ports Power Foam's activations, Steiner points and regular-triangulation builder, then runs the curved adjacency for a sweep of 12 curvatures per sign and ships the union of the edge sets (a superset is always safe: the first bisector crossed from inside a cell is a true face).
- A canonicalisation tool that applies one similarity transform to a whole checkpoint (points, radii through the softplus, densities, dipole frames, view-dependent axes, cameras) so COLMAP's arbitrary units become metres with the floor at zero, measured from the training cameras and a plane fit through the floor cells.
- Training on Vast.ai: 300k cells, 100k iterations, 32 minutes on a 4090, driven over SSH; the resulting 102 MB scene is shipped gzipped and decompressed in the browser.

## Challenges

- **Precision near flat.** Near $k=0$ every curved quantity is an $O(s^2)$ difference of $O(1)$ numbers; at $s = 10^{-3}$ the first version rendered noise (5 dB). Storing $a_0 - 1$ instead of $a_0$ and rewriting the ball test around the chord $-\tfrac12\langle o-p,\, o-p\rangle_\kappa$ brought the GPU to 45–66 dB of a float64 reference across the whole slider. Same equations, different order of operations.
- **Spheres are periodic.** A plane on $\mathbb{S}^3$ has infinitely many crossings; the walker had to be rewritten around "first exit after $t$ / last entry before $t$" instead of "the root".
- **Real captures are not synthetic rooms.** A trained foam is a thin shell of dense cells around every surface with nothing inside; our collision code walked straight through a bench until we swept a column of probes along each step and treated whole cell balls as solid. The ground is littered with ankle-height clutter cells. The first box universe placed its bottom face twelve centimetres above the floor, so every downward ray teleported forever.
- **Validating against the real thing.** Our cropped export scored 11 dB against Power Foam's own renders, and an afternoon went into discovering that the entire gap was the far sky and hills we had cropped away; the uncropped export scored 44 dB.

## What we learned

That a representation designed for differentiable rendering turns out to be the right representation for non-Euclidean rendering too, because its primitive operation, "which half-space am I in", is linear in the ambient space of every constant-curvature geometry. And that the hard part of a hackathon maths project is not the maths but the conditioning: the formulas were right on day one and the pictures were wrong for two more.

## References

Govindarajan, Rebain, Verbin, Yi, Prabhu, Tagliasacchi. *Power Foam: Unifying Real-Time Differentiable Ray Tracing and Rasterization.* arXiv:2604.24994, 2026. https://github.com/theialab/powerfoam · Barron et al., *Mip-NeRF 360*, CVPR 2022 (the Treehill capture) · Nielsen & Nock, *Hyperbolic Voronoi diagrams made easy*, 2010 · Weeks, *The Shape of Space*.
