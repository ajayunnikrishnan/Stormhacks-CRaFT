import { describe, expect, it } from "vitest";
import { type Kappa, v4, dot4, form, apply, mul, identity, embedPoint, geodesic, tangentialize, planeExitAfter, isometryError, distance } from "../src/geometry/space";
import { cubeHalfWidth, dihedralAngle, torus3, halfTurnSpace, kleinSpace, hyperbolicCubeHoneycomb, tesseractHoneycomb, flatCubeHoneycomb, insideDomain, domainExit, type Domain } from "../src/topology/domain";

function rng(seed: number) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const R = rng(5);

const ALL: () => Domain[] = () => [torus3([1, 0.7, 1.3]), halfTurnSpace([1, 1, 1]), kleinSpace([1, 1, 1]), flatCubeHoneycomb(1), hyperbolicCubeHoneycomb(), tesseractHoneycomb()];

/** Numerically measure the dihedral angle at a shared edge: the angle between the two faces'
 * inward tangents perpendicular to the edge, computed from points on the faces near the edge. */
function numericDihedral(d: Domain, i: number, j: number): number {
  const k = d.kappa;
  const fi = d.faces[i], fj = d.faces[j];
  // edge direction: common to both planes; find the point on the edge: midpoint of the two centres projected...
  // Simpler: pick the edge point as the normalised projective point where both planes and the plane
  // through the origin spanned by dir_i, dir_j meet: solve in the 2D (dir_i, dir_j) plane.
  const a = fi.dir, b = fj.dir;
  // parametrize x̄ = α a + β b with w_i·x = 0 = w_j·x (in projective coords y = x̄/x0)
  // For a box, planes are d·y = tanh/tan(h) etc.; solve the 2×2 linear system numerically
  const yi = projectiveOffset(k, fi.dist), yj = projectiveOffset(k, fj.dist);
  // d_i·y = yi, d_j·y = yj with y = α a + β b, a⊥b unit: α = yi, β = yj
  const y = [yi * a[0] + yj * b[0], yi * a[1] + yj * b[1], yi * a[2] + yj * b[2]];
  const edgePt = fromProjective(k, y);
  expect(Math.abs(dot4(fi.w, edgePt))).toBeLessThan(1e-9);
  expect(Math.abs(dot4(fj.w, edgePt))).toBeLessThan(1e-9);
  // inward tangent on face i at the edge point, perpendicular to the edge: move within plane i toward its centre
  const ti = tangentInPlaneToward(k, edgePt, fi.centre);
  const tj = tangentInPlaneToward(k, edgePt, fj.centre);
  return Math.acos(Math.max(-1, Math.min(1, form(k, ti, tj))));
}
function projectiveOffset(k: Kappa, h: number): number { return k < 0 ? Math.tanh(h) : k > 0 ? Math.tan(h) : h; }
function fromProjective(k: Kappa, y: number[]) {
  const n2 = y[0] * y[0] + y[1] * y[1] + y[2] * y[2];
  if (k === 0) return v4(1, y[0], y[1], y[2]);
  const x0 = k < 0 ? 1 / Math.sqrt(1 - n2) : 1 / Math.sqrt(1 + n2);
  return v4(x0, x0 * y[0], x0 * y[1], x0 * y[2]);
}
function tangentInPlaneToward(k: Kappa, x: ReturnType<typeof v4>, y: ReturnType<typeof v4>) {
  const d = distance(k, x, y);
  const u = v4();
  const c = k > 0 ? Math.cos(d) : k < 0 ? Math.cosh(d) : 1, s = k > 0 ? Math.sin(d) : k < 0 ? Math.sinh(d) : d;
  for (let i = 0; i < 4; i++) u[i] = (y[i] - c * x[i]) / s;
  return u;
}

describe("cube sizes from dihedral angles (§3.8)", () => {
  it("formula values", () => {
    expect(cubeHalfWidth(-1, 72)).toBeCloseTo(0.5306375309525179, 9);
    expect(cubeHalfWidth(1, 120)).toBeCloseTo(Math.PI / 4, 12);
    expect(cubeHalfWidth(0, 90, 2)).toBe(2);
  });
  it.each([["{4,3,4}", flatCubeHoneycomb(1), 90], ["{4,3,5}", hyperbolicCubeHoneycomb(), 72], ["{4,3,3}", tesseractHoneycomb(), 120]] as const)("%s: numerically measured dihedral angle = %i°", (_n, d, deg) => {
    // adjacent face pairs: (+x,+y), (+x,−z), (−y,+z) ...
    for (const [i, j] of [[0, 2], [0, 5], [3, 4], [1, 2], [4, 0]]) {
      expect((dihedralAngle(d.kappa, d.faces[i], d.faces[j]) * 180) / Math.PI).toBeCloseTo(deg, 9);
      expect((numericDihedral(d, i, j) * 180) / Math.PI).toBeCloseTo(deg, 6);
    }
  });
});

describe("face pairings (§3.8 tests)", () => {
  for (const d of ALL())
    it(`${d.name}: g_f maps face f onto its partner exactly, g_partner ∘ g_f = id, isometries valid`, () => {
      const k = d.kappa;
      for (let f = 0; f < d.faces.length; f++) {
        const F = d.faces[f], P = d.faces[F.partner];
        expect(isometryError(k, F.g)).toBeLessThan(1e-12);
        const comp = mul(P.g, F.g), I = identity();
        for (let i = 0; i < 16; i++) expect(Math.abs(comp[i] - I[i])).toBeLessThan(1e-12);
        // sample points on face f (within the face's chart) and map them
        for (let n = 0; n < 50; n++) {
          const u = (R() - 0.5) * 1.6 * F.dist, w2 = (R() - 0.5) * 1.6 * F.dist;
          const x = apply(F.frame, embedPoint(k, [u, w2, 0])); // point in the face plane
          expect(Math.abs(dot4(F.w, x))).toBeLessThan(1e-9);
          const y = apply(F.g, x);
          expect(Math.abs(dot4(P.w, y))).toBeLessThan(1e-9); // lands on the partner plane
          // the outward normal of f maps to the INWARD normal of the partner (ray re-enters)
          const nOut = apply(F.g, F.normal);
          expect(form(k, nOut, P.normal)).toBeLessThan(-0.999);
        }
      }
    });
  it("domain centre is inside; a ray from the centre exits through exactly the face it points at", () => {
    for (const d of ALL()) {
      const k = d.kappa;
      expect(insideDomain(d, v4(1, 0, 0, 0))).toBe(true);
      for (let f = 0; f < 6; f++) {
        const dir = d.faces[f].dir;
        const o = v4(1, 0, 0, 0), v = v4(0, dir[0], dir[1], dir[2]);
        const ex = domainExit(d, o, v, 0, planeExitAfter)!;
        expect(ex.face).toBe(f);
        expect(ex.t).toBeCloseTo(d.faces[f].dist, 9);
      }
    }
  });
  it("transported rays stay consistent: exit, map, continue, and the point stays inside the domain", () => {
    for (const d of ALL()) {
      const k = d.kappa;
      for (let n = 0; n < 100; n++) {
        let o = embedPoint(k, [(R() - 0.5) * 0.5 * d.inradius, (R() - 0.5) * 0.5 * d.inradius, (R() - 0.5) * 0.5 * d.inradius]);
        let v = tangentialize(k, o, v4(0, R() - 0.5, R() - 0.5, R() - 0.5));
        for (let hop = 0; hop < 20; hop++) {
          const ex = domainExit(d, o, v, 1e-9, planeExitAfter);
          if (!ex || !Number.isFinite(ex.t)) break;
          const x = geodesic(k, o, v, ex.t);
          const g = d.faces[ex.face].g;
          const o2 = apply(g, x);
          const v2 = apply(g, (() => { const dv = v4(); const c = k > 0 ? Math.cos(ex.t) : k < 0 ? Math.cosh(ex.t) : 1, s = k > 0 ? Math.sin(ex.t) : k < 0 ? Math.sinh(ex.t) : ex.t; for (let i = 0; i < 4; i++) dv[i] = -k * s * o[i] + c * v[i]; return dv; })());
          expect(insideDomain(d, o2, 1e-7)).toBe(true);
          // after a tiny step inward it is strictly inside
          const probe = geodesic(k, o2, v2, 1e-4);
          expect(insideDomain(d, probe, 1e-9)).toBe(true);
          o = o2; v = v2;
        }
      }
    }
  });
});
