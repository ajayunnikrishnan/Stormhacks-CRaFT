import { describe, expect, it } from "vitest";
import { dot4, apply, mul, identity, isometryError, form, type V4 } from "../src/geometry/space";
import { dihedralAngle, insideDomain, type Domain } from "../src/topology/domain";
import { dodecahedronInradius, poincareDodecahedralSpace, seifertWeberSpace, dodecahedronVertices, dodecahedralDomain } from "../src/topology/dodecahedron";

const near = (a: V4, b: V4, tol = 1e-7) => Math.abs(a[0] - b[0]) < tol && Math.abs(a[1] - b[1]) < tol && Math.abs(a[2] - b[2]) < tol && Math.abs(a[3] - b[3]) < tol;

/** Edge cycle test: for every edge (v1,v2) shared by faces (f, f'), follow the pairings around the
 * edge; the composed isometry must be the identity after exactly 2π/φ steps. */
function edgeCycles(d: Domain): { lengths: number[]; maxErr: number } {
  const verts = dodecahedronVertices(d);
  const onFace = (f: number, x: V4) => Math.abs(dot4(d.faces[f].w, x)) < 1e-7;
  const facesOf = (x: V4) => d.faces.map((_, f) => f).filter((f) => onFace(f, x));
  // edges: vertex pairs sharing exactly two faces
  const edges: [number, number, number, number][] = []; // v1, v2, fA, fB
  for (let i = 0; i < verts.length; i++)
    for (let j = i + 1; j < verts.length; j++) {
      const common = facesOf(verts[i]).filter((f) => facesOf(verts[j]).includes(f));
      if (common.length === 2) edges.push([i, j, common[0], common[1]]);
    }
  expect(edges.length).toBe(30);
  const lengths: number[] = [];
  let maxErr = 0;
  for (const [vi, vj, fA, fB] of edges) {
    // start: we stand inside the domain next to the edge, between faces fA and fB; cross fA.
    let M = identity();
    let a = verts[vi], b = verts[vj];
    let cross = fA, other = fB;
    let steps = 0;
    for (; steps < 20; steps++) {
      const g = d.faces[cross].g;
      M = mul(g, M);
      a = apply(g, a); b = apply(g, b);
      // we arrive on partner(cross); the edge (a,b) is shared by partner(cross) and one more face
      const p = d.faces[cross].partner;
      const fa = facesOf(a).filter((f) => facesOf(b).includes(f));
      expect(fa.length).toBe(2);
      expect(fa.includes(p)).toBe(true);
      const nextCross = fa.find((f) => f !== p)!;
      other = p;
      cross = nextCross;
      // closed when the edge is back in place and M is the identity
      if (near(a, verts[vi]) && near(b, verts[vj])) {
        let err = 0;
        const I = identity();
        for (let q = 0; q < 16; q++) err = Math.max(err, Math.abs(M[q] - I[q]));
        if (err < 1e-6) { steps++; break; }
      }
    }
    lengths.push(steps);
    const I = identity();
    for (let q = 0; q < 16; q++) maxErr = Math.max(maxErr, Math.abs(M[q] - I[q]));
    void other;
  }
  return { lengths, maxErr };
}

describe("dodecahedral spaces", () => {
  it("inradii: π/10 for the 120-cell dodecahedron (S³, 120°), 0.996 for Seifert–Weber (H³, 72°)", () => {
    expect(dodecahedronInradius(1, 120)).toBeCloseTo(Math.PI / 10, 10);
    expect(dodecahedronInradius(-1, 72)).toBeCloseTo(0.9962, 3);
  });
  for (const [d, deg] of [[poincareDodecahedralSpace(), 120], [seifertWeberSpace(), 72]] as const)
    it(`${d.name}: 12 faces, dihedral ${deg}°, pairings valid, vertices on 3 faces, centre inside`, () => {
      expect(d.faces.length).toBe(12);
      const verts = dodecahedronVertices(d);
      expect(verts.length).toBe(20);
      for (const v of verts) {
        const on = d.faces.filter((f) => Math.abs(dot4(f.w, v)) < 1e-7);
        expect(on.length).toBe(3);
        expect(insideDomain(d, v, 1e-7)).toBe(true);
        // adjacent faces at this vertex have the right dihedral angle
        expect((dihedralAngle(d.kappa, on[0], on[1]) * 180) / Math.PI).toBeCloseTo(deg, 7);
      }
      for (const f of d.faces) {
        expect(isometryError(d.kappa, f.g)).toBeLessThan(1e-12);
        const comp = mul(d.faces[f.partner].g, f.g), I = identity();
        for (let i = 0; i < 16; i++) expect(Math.abs(comp[i] - I[i])).toBeLessThan(1e-12);
        const img = apply(f.g, f.centre);
        expect(Math.abs(dot4(d.faces[f.partner].w, img))).toBeLessThan(1e-9);
        expect(form(d.kappa, apply(f.g, f.normal), d.faces[f.partner].normal)).toBeLessThan(-0.999);
      }
    });
  it("edge cycles close with the identity: 3 around each edge for PDS (120°), 5 for Seifert–Weber (72°)", () => {
    const p = edgeCycles(poincareDodecahedralSpace());
    expect(p.maxErr).toBeLessThan(1e-8);
    expect(p.lengths.every((l) => l === 3)).toBe(true);
    const s = edgeCycles(seifertWeberSpace());
    expect(s.maxErr).toBeLessThan(1e-8);
    expect(s.lengths.every((l) => l === 5)).toBe(true);
  });
  it("the wrong twist does NOT close (sanity of the test): 36° twist in H³ / 108° in S³ fail", () => {
    const bad1 = dodecahedralDomain("x", "x", -1, 72, Math.PI / 5, "");
    const bad2 = dodecahedralDomain("y", "y", 1, 120, (3 * Math.PI) / 5, "");
    let threw = 0;
    for (const b of [bad1, bad2]) {
      try { const r = edgeCycles(b); if (r.maxErr > 1e-6 || !r.lengths.every((l) => l === (b.kappa < 0 ? 5 : 3))) threw++; } catch { threw++; }
    }
    expect(threw).toBe(2);
  });
});
