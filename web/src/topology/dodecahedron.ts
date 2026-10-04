/**
 * Regular dodecahedral domains (§3.8, closed manifolds):
 *   Poincaré dodecahedral space  S³: dihedral 120°, opposite faces glued with a  36° (π/5) twist
 *   Seifert–Weber space          H³: dihedral  72°, opposite faces glued with a 108° (3π/5) twist
 * Face poles are the 12 icosahedron vertex directions. For adjacent faces at angular
 * separation θ = atan 2 between their directions:
 *   κ=−1: cos φ = sinh²h − cosh²h cos θ   ⇒  sinh²h = (cos φ + cos θ)/(1 − cos θ)
 *   κ=+1: cos φ = −sin²h − cos²h cos θ    ⇒  sin²h  = (−cos φ − cos θ)/(1 − cos θ)
 * giving h = π/10 for the 120-cell's dodecahedron (S³) — a known value — and h ≈ 0.996 (H³).
 * Correct twist sense is fixed by the edge-cycle condition (tests/dodecahedron.test.ts):
 * going around any edge through the pairings composes to the identity.
 */
import { type Kappa, type M4, type V4, v4, mul, inverse, identity, translationTo, rotation, translationByVector, cs, sn, apply, logAtOrigin } from "../geometry/space";
import { polyhedronVertices, type Domain, type Face } from "./domain";

const PHI = (1 + Math.sqrt(5)) / 2;

/** 12 unit directions of icosahedron vertices (0, ±1, ±φ) and cyclic permutations, as 6 antipodal pairs. */
export function icosahedronDirections(): number[][] {
  const raw: number[][] = [];
  for (const s1 of [1, -1]) for (const s2 of [1, -1]) {
    raw.push([0, s1, s2 * PHI]);
    raw.push([s1, s2 * PHI, 0]);
    raw.push([s2 * PHI, 0, s1]);
  }
  const n = raw.map((d) => { const l = Math.hypot(...d); return [d[0] / l, d[1] / l, d[2] / l]; });
  // order as antipodal pairs (2i, 2i+1)
  const used = new Array(12).fill(false);
  const out: number[][] = [];
  for (let i = 0; i < 12; i++) {
    if (used[i]) continue;
    const j = n.findIndex((d, jj) => !used[jj] && jj !== i && Math.abs(d[0] + n[i][0]) < 1e-9 && Math.abs(d[1] + n[i][1]) < 1e-9 && Math.abs(d[2] + n[i][2]) < 1e-9);
    used[i] = used[j] = true;
    out.push(n[i], n[j]);
  }
  return out;
}

export function dodecahedronInradius(k: Kappa, dihedralDeg: number): number {
  const cphi = Math.cos((dihedralDeg * Math.PI) / 180);
  const cth = 1 / Math.sqrt(5); // cos(atan 2) = angle between adjacent icosahedron vertices
  if (k < 0) return Math.asinh(Math.sqrt((cphi + cth) / (1 - cth)));
  if (k > 0) return Math.asin(Math.sqrt((-cphi - cth) / (1 - cth)));
  throw new Error("flat dodecahedra do not tile");
}

function faceFromPole(k: Kappa, dir: number[], h: number): { w: V4; centre: V4; normal: V4 } {
  const d = dir;
  const centre = v4(cs(k, h), sn(k, h) * d[0], sn(k, h) * d[1], sn(k, h) * d[2]);
  const normal = k < 0 ? v4(Math.sinh(h), Math.cosh(h) * d[0], Math.cosh(h) * d[1], Math.cosh(h) * d[2]) : v4(-Math.sin(h), Math.cos(h) * d[0], Math.cos(h) * d[1], Math.cos(h) * d[2]);
  return { w: v4(k * normal[0], normal[1], normal[2], normal[3]), centre, normal };
}

/** Rotation about unit axis u by angle a (column-major 3×3 → mat4 block). */
export function rotationAbout(u: number[], a: number): M4 {
  const c = Math.cos(a), s = Math.sin(a), C = 1 - c;
  const [x, y, z] = u;
  return rotation([c + x * x * C, y * x * C + z * s, z * x * C - y * s, x * y * C - z * s, c + y * y * C, z * y * C + x * s, x * z * C + y * s, y * z * C - x * s, c + z * z * C]);
}

function frameFor(k: Kappa, dir: number[], centre: V4): M4 {
  const z = dir;
  const a = Math.abs(z[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const x = normalize(cross(a, z));
  const y = cross(z, x);
  return mul(translationTo(k, centre), rotation([x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]]));
}

/**
 * Dodecahedral domain with opposite faces glued by: translate across (−2h along the face axis)
 * after rotating about the axis by `twist` (radians, right-handed about the +face direction).
 */
export function dodecahedralDomain(id: string, name: string, k: Kappa, dihedralDeg: number, twist: number, hint: string): Domain {
  const h = dodecahedronInradius(k, dihedralDeg);
  const dirs = icosahedronDirections();
  const faces: Face[] = dirs.map((dir, i) => {
    const { w, centre, normal } = faceFromPole(k, dir, h);
    const frame = frameFor(k, dir, centre);
    return { w, centre, normal, dir, dist: h, partner: i ^ 1, g: identity(), frame, invFrame: inverse(k, frame) };
  });
  for (let i = 0; i < 12; i += 2) {
    const d = dirs[i];
    const g = mul(translationByVector(k, [-2 * h * d[0], -2 * h * d[1], -2 * h * d[2]]), rotationAbout(d, twist));
    faces[i].g = g;
    faces[i + 1].g = inverse(k, g);
  }
  const vertices = polyhedronVertices(k, faces, 1 / Math.sqrt(5));
  return { id, name, kappa: k, orientable: true, faces, inradius: h, hint, vertices };
}

export function poincareDodecahedralSpace(): Domain {
  return dodecahedralDomain("pds", "Poincaré dodecahedral space (S³)", 1, 120, Math.PI / 5, "twelve pentagonal doors; step through one and you come back turned by 36°");
}
export function seifertWeberSpace(): Domain {
  return dodecahedralDomain("sw", "Seifert–Weber space (H³)", -1, 72, (3 * Math.PI) / 5, "a hyperbolic dodecahedron glued with a 108° twist");
}

// ---------------------------------------------------------------- geometry for tests
export function dodecahedronVertices(d: Domain): V4[] { return d.vertices; }

function cross(a: number[], b: number[]) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function normalize(a: number[]) { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; }
export const _u = { apply, logAtOrigin };
