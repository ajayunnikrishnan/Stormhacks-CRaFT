/**
 * Fundamental domains and face pairings (§3.8).
 *
 * A space is a convex fundamental polyhedron D = { x : w_f·x ≤ 0 ∀ faces f } (covectors as in
 * geometry/space.ts) together with, for each face f, an isometry g_f that maps face f onto its
 * partner face π(f) so that g_{π(f)} = g_f⁻¹. A ray leaving D through f continues as g_f(ray).
 *
 * Face planes are described by their pole: the point c_f at distance h_f from the origin along
 * the unit direction d_f (the face centre), with outward unit normal n_f there:
 *   κ=−1: c = (cosh h, sinh h d),  n = (sinh h, cosh h d)
 *   κ=+1: c = (cos h,  sin h d),   n = (−sin h, cos h d)
 *   κ= 0: c = (1, h d),            plane d·x̄ = h  ⇒  w = (−h, d)
 * and the covector is w = J n (κ≠0). Dihedral angle between faces i, j (outward normals):
 *   cos φ = −⟨n_i, n_j⟩_κ, so a cube with face distance h has cos φ = κ'·… ; see cubeHalfWidth().
 */
import {
  type Kappa, type M4, type V4, v4, form, dot4, embedPoint, translationByVector, rotation, mul, inverse,
  identity, apply, ORIGIN, translationTo, cs, sn,
} from "../geometry/space";
import { poincareDodecahedralSpace, seifertWeberSpace } from "./dodecahedron";

export interface Face {
  w: V4; // covector: inside ⇔ w·x ≤ 0
  centre: V4; // point on the face (model coords)
  normal: V4; // outward unit tangent normal at centre
  dir: number[]; // unit direction from the origin (for building frames)
  dist: number; // geodesic distance origin → face
  partner: number;
  g: M4; // isometry applied to a ray exiting through this face (maps f onto partner)
  frame: M4; // isometry origin → face centre with e3 ↦ outward normal (for the lookup chart)
  invFrame: M4;
}

export interface Domain {
  id: string;
  name: string;
  kappa: Kappa;
  orientable: boolean;
  faces: Face[];
  inradius: number; // model units
  hint: string;
  vertices: V4[]; // polyhedron vertices (model coords), used to size the point-location charts
}

/** Vertices as triple intersections of face planes that are pairwise adjacent (angle between
 * directions = adjAngle) — in projective coordinates the planes are linear: d_f·y = off_f. */
export function polyhedronVertices(k: Kappa, faces: { dir: number[]; dist: number }[], adjCos: number): V4[] {
  const off = (h: number) => (k < 0 ? Math.tanh(h) : k > 0 ? Math.tan(h) : h);
  const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const adj = (a: number, b: number) => Math.abs(dot(faces[a].dir, faces[b].dir) - adjCos) < 1e-9;
  const out: V4[] = [];
  const n = faces.length;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let l = j + 1; l < n; l++) {
    if (!(adj(i, j) && adj(j, l) && adj(i, l))) continue;
    const y = solve3([faces[i].dir, faces[j].dir, faces[l].dir], [off(faces[i].dist), off(faces[j].dist), off(faces[l].dist)]);
    const n2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
    const x0 = k === 0 ? 1 : k < 0 ? 1 / Math.sqrt(1 - n2) : 1 / Math.sqrt(1 + n2);
    out.push(v4(x0, x0 * y[0], x0 * y[1], x0 * y[2]));
  }
  return out;
}
export function solve3(A: number[][], b: number[]): number[] {
  const [a, c, e] = A;
  const det = a[0] * (c[1] * e[2] - c[2] * e[1]) - a[1] * (c[0] * e[2] - c[2] * e[0]) + a[2] * (c[0] * e[1] - c[1] * e[0]);
  const det1 = b[0] * (c[1] * e[2] - c[2] * e[1]) - a[1] * (b[1] * e[2] - c[2] * b[2]) + a[2] * (b[1] * e[1] - c[1] * b[2]);
  const det2 = a[0] * (b[1] * e[2] - c[2] * b[2]) - b[0] * (c[0] * e[2] - c[2] * e[0]) + a[2] * (c[0] * b[2] - b[1] * e[0]);
  const det3 = a[0] * (c[1] * b[2] - b[1] * e[1]) - a[1] * (c[0] * b[2] - b[1] * e[0]) + b[0] * (c[0] * e[1] - c[1] * e[0]);
  return [det1 / det, det2 / det, det3 / det];
}

function faceFromPole(k: Kappa, dir: number[], h: number): { w: V4; centre: V4; normal: V4 } {
  const d = dir;
  let centre: V4, normal: V4;
  if (k === 0) {
    centre = v4(1, h * d[0], h * d[1], h * d[2]);
    normal = v4(0, d[0], d[1], d[2]);
    return { w: v4(-h, d[0], d[1], d[2]), centre, normal };
  }
  centre = v4(cs(k, h), sn(k, h) * d[0], sn(k, h) * d[1], sn(k, h) * d[2]);
  normal = k < 0 ? v4(Math.sinh(h), Math.cosh(h) * d[0], Math.cosh(h) * d[1], Math.cosh(h) * d[2]) : v4(-Math.sin(h), Math.cos(h) * d[0], Math.cos(h) * d[1], Math.cos(h) * d[2]);
  const w = v4(k * normal[0], normal[1], normal[2], normal[3]); // J n
  return { w, centre, normal };
}

/** Frame at the face centre: translation to the centre composed with a rotation taking e3 → dir. */
function faceFrame(k: Kappa, dir: number[], centre: V4): M4 {
  // rotation R3 with third column = dir, others orthonormal
  const z = dir;
  const a = Math.abs(z[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const x = normalize(cross(a, z));
  const y = cross(z, x);
  const R3 = [x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]]; // column-major
  return mul(translationTo(k, centre), rotation(R3));
}

/**
 * Face distance h of a regular cube with interior dihedral angle φ (prompt §3.8):
 *   κ=−1: cos φ = sinh² h   κ=+1: cos φ = −sin² h   κ=0: φ = 90°, h free.
 * {4,3,4}: 90°, {4,3,5}: 72° ⇒ h = asinh(√cos72°) = 0.5306, {4,3,3}: 120° ⇒ h = π/4.
 */
export function cubeHalfWidth(k: Kappa, dihedralDeg: number, flatHalfWidth = 1): number {
  const c = Math.cos((dihedralDeg * Math.PI) / 180);
  if (k < 0) { if (c <= 0) throw new Error("hyperbolic cube needs dihedral < 90°"); return Math.asinh(Math.sqrt(c)); }
  if (k > 0) { if (c >= 0) throw new Error("spherical cube needs dihedral > 90°"); return Math.asin(Math.sqrt(-c)); }
  return flatHalfWidth;
}

/** Interior dihedral angle between two faces from their outward normals: cos φ = −⟨n_i,n_j⟩. */
export function dihedralAngle(k: Kappa, a: Face, b: Face): number {
  return Math.acos(Math.max(-1, Math.min(1, -form(k, a.normal, b.normal))));
}

const AXES = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];

/**
 * Box domain with the six axis faces at distances (hx, hy, hz) and per-pair gluing maps.
 * `twist(pairIndex)` returns an extra isometry (about the face axis) composed into the pairing of
 * the +face: g_{+} = T(−2h d)·twist, g_{−} = g_{+}⁻¹.
 */
export function boxDomain(id: string, name: string, k: Kappa, half: [number, number, number], opts: { twist?: (axis: number) => M4 | null; orientable?: boolean; hint?: string } = {}): Domain {
  const faces: Face[] = [];
  for (let i = 0; i < 6; i++) {
    const dir = AXES[i];
    const axis = i >> 1;
    const h = half[axis];
    const { w, centre, normal } = faceFromPole(k, dir, h);
    const frame = faceFrame(k, dir, centre);
    faces.push({ w, centre, normal, dir, dist: h, partner: i ^ 1, g: identity(), frame, invFrame: inverse(k, frame) });
  }
  for (let axis = 0; axis < 3; axis++) {
    const plus = 2 * axis, minus = plus + 1;
    const h = half[axis];
    const d = AXES[plus];
    // exiting through +face: move the ray back by 2h along the axis so it re-enters at −face
    let g = translationByVector(k, [-2 * h * d[0], -2 * h * d[1], -2 * h * d[2]]);
    const tw = opts.twist?.(axis);
    if (tw) g = mul(g, tw); // apply the twist about the axis (fixes the origin), then translate
    faces[plus].g = g;
    faces[minus].g = inverse(k, g);
  }
  const vertices = polyhedronVertices(k, faces, 0); // box: adjacent faces are perpendicular
  return { id, name, kappa: k, orientable: opts.orientable ?? true, faces, inradius: Math.min(...half), hint: opts.hint ?? "", vertices };
}

function rotAbout(axis: number, ang: number): M4 {
  const c = Math.cos(ang), s = Math.sin(ang);
  const R = axis === 0 ? [1, 0, 0, 0, c, s, 0, -s, c] : axis === 1 ? [c, 0, -s, 0, 1, 0, s, 0, c] : [c, s, 0, -s, c, 0, 0, 0, 1];
  return rotation(R);
}
function reflectAcross(axis: number): M4 {
  const R = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  R[axis * 4] = -1;
  return rotation(R);
}

// ---------------------------------------------------------------- catalogue
export function torus3(half: [number, number, number]): Domain {
  return boxDomain("torus3", "3-torus", 0, half, { hint: "walk through any wall and come back from the opposite one, unchanged" });
}
export function halfTurnSpace(half: [number, number, number]): Domain {
  // the ±z pair is glued with a 180° turn about the z axis
  return boxDomain("halfturn", "half-turn space", 0, half, { twist: (a) => (a === 2 ? rotAbout(2, Math.PI) : null), hint: "go through the far wall: you come back rotated 180°" });
}
export function kleinSpace(half: [number, number, number]): Domain {
  // non-orientable: the ±x pair is glued with a reflection in the (x, y) plane's normal… i.e. z ↦ −z
  return boxDomain("klein", "Klein space (non-orientable)", 0, half, { twist: (a) => (a === 0 ? reflectAcross(2) : null), orientable: false, hint: "through the side wall, you come back mirrored" });
}
export function hyperbolicCubeHoneycomb(): Domain {
  const h = cubeHalfWidth(-1, 72);
  return boxDomain("h435", "{4,3,5} hyperbolic cube honeycomb", -1, [h, h, h], { hint: "five cubes meet at every edge: look along one and count" });
}
export function flatCubeHoneycomb(half: number): Domain {
  return boxDomain("e434", "{4,3,4} cubic tiling", 0, [half, half, half], { hint: "four cubes meet at every edge" });
}
export function tesseractHoneycomb(): Domain {
  const h = cubeHalfWidth(1, 120); // π/4
  return boxDomain("s433", "{4,3,3} tesseract (8 cubes tile S³)", 1, [h, h, h], { hint: "three cubes at every edge; eight cubes fill the whole universe" });
}

export const DOMAIN_IDS = ["none", "torus3", "halfturn", "klein", "e434", "h435", "s433", "pds", "sw"] as const;
export type DomainId = (typeof DOMAIN_IDS)[number];

/** Build a domain by id. `flatHalf` sizes the flat boxes (model units = metres when κ=0). */
export function makeDomain(id: DomainId, flatHalf: [number, number, number]): Domain | null {
  switch (id) {
    case "none": return null;
    case "torus3": return torus3(flatHalf);
    case "halfturn": return halfTurnSpace(flatHalf);
    case "klein": return kleinSpace(flatHalf);
    case "e434": return flatCubeHoneycomb(Math.max(...flatHalf));
    case "h435": return hyperbolicCubeHoneycomb();
    case "s433": return tesseractHoneycomb();
    case "pds": return poincareDodecahedralSpace();
    case "sw": return seifertWeberSpace();
  }
}

/** First domain face the ray exits after arc length tMin, or null. */
export function domainExit(d: Domain, o: V4, v: V4, tMin: number, planeExitAfter: (k: Kappa, A: number, B: number, tMin: number) => number): { face: number; t: number } | null {
  let best = -1, bt = Infinity;
  for (let f = 0; f < d.faces.length; f++) {
    const w = d.faces[f].w;
    const t = planeExitAfter(d.kappa, dot4(w, o), dot4(w, v), tMin);
    if (t < bt) { bt = t; best = f; }
  }
  return best < 0 ? null : { face: best, t: bt };
}

export function insideDomain(d: Domain, x: V4, tol = 1e-9): boolean {
  return d.faces.every((f) => dot4(f.w, x) <= tol);
}

// helpers
function cross(a: number[], b: number[]) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function normalize(a: number[]) { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; }
export const _unused = { embedPoint, apply, ORIGIN };
