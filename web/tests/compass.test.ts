/**
 * Compass holonomy (§4 Tools, §7 Gauss–Bonnet): walk a closed geodesic polygon on the eye
 * plane by alternating straight moves and turns by the exterior angles; back at the start with
 * the original heading, the compass must be off by the holonomy angle, i.e. the angle excess
 * κ·Area. For a geodesic triangle with two right angles we build it directly: go out along x,
 * turn, go along a geodesic to a third point, turn, come back. We compute the exterior angles
 * numerically from the geometry (Eq. 5 tangents) and compare the compass angle to κ·A where
 * A comes from Gauss–Bonnet: A = κ(Σ interior − π) = κ(2π − Σ exterior − π)... i.e. we check
 * compass rotation == 2π − Σ exterior (the walker's total turning minus a full turn).
 */
import { describe, expect, it } from "vitest";
import { type Kappa, embedPoint, tangentToward, form, apply, inverse, ORIGIN, v4, distance } from "../src/geometry/space";
import { IsoCamera } from "../src/game/isocamera";
import { Player } from "../src/game/player";

function fakeScene() {
  // minimal SceneArrays: one cell so Player constructs; collisions are disabled in the test
  return { n: 1, k: 1, d: 1, pos: new Float32Array([0, 0, 0, 1]), nsigma: new Uint16Array(4), siteoff: new Uint16Array(4), svaxis: new Uint16Array(3), svrgb: new Uint16Array(3), adjOff: new Uint32Array([0, 0]), adjIdx: new Uint32Array(0), manifest: {} as never };
}

/** Exterior turning angle at corner B of the geodesic path A→B→C: angle between incoming and outgoing tangents at B (signed, + = left turn seen from above with y up... we use magnitude and consistent orientation). */
function exteriorAngle(k: Kappa, A: Float64Array, B: Float64Array, C: Float64Array): number {
  const inDir = (() => { const { u } = tangentToward(k, A, B); /* transport to B: tangent at B of the geodesic from A */ const d = distance(k, A, B); const c = k > 0 ? Math.cos(d) : k < 0 ? Math.cosh(d) : 1, s = k > 0 ? Math.sin(d) : k < 0 ? Math.sinh(d) : d; const out = v4(); for (let i = 0; i < 4; i++) out[i] = -k * s * A[i] + c * u[i]; return out; })();
  const outDir = tangentToward(k, B, C).u;
  const cosA = Math.max(-1, Math.min(1, form(k, inDir, outDir)));
  // sign from the eye-plane normal (x2 axis): cross product's y component in the local frame
  const sx = inDir[1] * outDir[3] - inDir[3] * outDir[1];
  return Math.sign(-sx) * Math.acos(cosA); // + = turn left (counter-clockwise seen from +y)
}

describe("compass holonomy around a geodesic triangle", () => {
  for (const k of [-1, 0, 1] as Kappa[])
    it(`κ=${k}: compass rotation == 2π − Σ exterior angles (= κ·area)`, () => {
      const side = k === 0 ? 1 : 0.9;
      // triangle on the eye plane (x, z): A = origin, B = side along −z (ahead), C = side along +x from A... make it general
      const A = ORIGIN;
      const B = embedPoint(k, [0, 0, -side]);
      const C = embedPoint(k, [0.8 * side, 0, -0.3 * side]);
      const cam = new IsoCamera();
      cam.setPose(k, A, 0); // looking down −z
      const player = new Player(fakeScene() as never);
      player.collisions = false;
      let totalTurn = 0;
      const legs: [Float64Array, Float64Array, Float64Array][] = [[C, A, B], [A, B, C], [B, C, A]]; // (prev, at, next)
      // start heading along A→B (−z): matches yaw 0
      const walkTo = (from: Float64Array, to: Float64Array) => {
        const d = distance(k, from, to);
        // the body heading already points at `to`; move straight ahead in steps
        const n = 50;
        for (let i = 0; i < n; i++) cam.moveBy([0, 0, -d / n]);
      };
      const turnTo = (prev: Float64Array, at: Float64Array, next: Float64Array) => {
        const ext = exteriorAngle(k, prev, at, next);
        // yawBy(a) turns the view right by a; a left turn by ext means yawBy(−ext)
        const R = (() => { const c = Math.cos(-ext), s = Math.sin(-ext); return [c, 0, -s, 0, 1, 0, s, 0, c]; })();
        cam.yawBy(-ext);
        const R4 = new Float64Array(16); R4[0] = 1; for (let c2 = 0; c2 < 3; c2++) for (let r = 0; r < 3; r++) R4[4 * (c2 + 1) + 1 + r] = R[3 * c2 + r];
        player.onYaw(R4);
        totalTurn += ext;
      };
      walkTo(A, B); turnTo(A, B, C);
      walkTo(B, C); turnTo(B, C, A);
      walkTo(C, A); turnTo(C, A, B);
      // back at A with the original heading?
      const pos = apply(inverse(k, cam.Wb), ORIGIN);
      expect(distance(k, pos, A)).toBeLessThan(2e-3);
      const heading = apply(inverse(k, cam.Wb), v4(0, 0, 0, -1));
      const { u: toB } = tangentToward(k, pos, B);
      expect(form(k, heading, toB)).toBeGreaterThan(0.9999);
      // holonomy: the compass (which only followed the yaws) is off from the heading by 2π − Σext
      const expected = 2 * Math.PI - totalTurn; // = κ·Area by Gauss–Bonnet
      const got = player.compassAngle();
      const wrap = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
      expect(Math.abs(wrap(got) - wrap(-expected)) < 2e-3 || Math.abs(wrap(got) - wrap(expected)) < 2e-3).toBe(true);
      if (k === 0) expect(Math.abs(wrap(got))).toBeLessThan(2e-3);
      else expect(Math.abs(expected)).toBeGreaterThan(0.05); // curved: non-trivial holonomy
      // and the magnitude equals κ·Area for a triangle: Area = κ(Σ interior − π) ⇒ compare with Σ interior
      const interior = legs.map(([p, at, n]) => Math.PI - Math.abs(exteriorAngle(k, p, at, n))).reduce((a, b) => a + b, 0);
      const areaTimesK = interior - Math.PI; // = κ·A
      expect(Math.abs(Math.abs(wrap(got)) - Math.abs(areaTimesK))).toBeLessThan(2e-3);
    });
});
