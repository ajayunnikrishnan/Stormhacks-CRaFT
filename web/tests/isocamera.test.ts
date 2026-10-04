import { describe, expect, it } from "vitest";
import { IsoCamera } from "../src/game/isocamera";
import { apply, v4, mul, rotation, inverse, translationTo, embedPoint, form, type Kappa } from "../src/geometry/space";

/** Exact repo-convention cameras (as tools/synth_scene.py builds them): right/up from eye + forward. */
function makeCam(name: string, eye: number[], fwd: number[], fovDeg = 70, aspect = 4 / 3) {
  const f = norm3(fwd);
  const r = norm3(cross3(f, [0, 1, 0]));
  const u = cross3(r, f);
  const ty = Math.tan((fovDeg * Math.PI) / 360), tx = ty * aspect;
  return { name, eye, right: r.map((v) => v * tx), up: u.map((v) => v * ty), width: 320, height: 240 };
}
const cams = [makeCam("corner", [-3, 1.6, 3], [1, -0.15, -1]), makeCam("box", [0, 1.4, 0.5], [0.8, -0.3, -1]), makeCam("up", [1, 1.6, -2], [-0.3, 0.5, 0.4])];
function norm3(a: number[]) { const l = Math.hypot(...a); return a.map((x) => x / l); }
function cross3(a: number[], b: number[]) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }

describe("IsoCamera.setFromRepoCamera", () => {
  for (const k of [-1, 0, 1] as Kappa[])
    it(`κ=${k}: W equals the direct rotation·T⁻¹ build, body forward is horizontal`, () => {
      for (const c of cams) {
        const cam = new IsoCamera();
        const centre = [0, 1.6, 0], s = k === 0 ? 1 : 0.2;
        cam.setFromRepoCamera(c, k, s, centre);
        const r = norm3(c.right), u = norm3(c.up), f = norm3(cross3(u, r));
        const R3 = [r[0], u[0], -f[0], r[1], u[1], -f[1], r[2], u[2], -f[2]];
        const pos = embedPoint(k, [(c.eye[0] - centre[0]) * s, (c.eye[1] - centre[1]) * s, (c.eye[2] - centre[2]) * s]);
        const Wdirect = mul(rotation(R3), inverse(k, translationTo(k, pos)));
        const W = cam.W();
        for (let i = 0; i < 16; i++) expect(Math.abs(W[i] - Wdirect[i])).toBeLessThan(1e-9);
        // body forward (−z in body coords) is orthogonal to the vertical at the camera position
        // (the transported e2; in curved space the tangent space off the eye plane is tilted, so the
        // ambient x2 component alone is not the right test)
        const fwdWorld = apply(inverse(k, cam.Wb), v4(0, 0, 0, -1));
        const vertical = apply(translationTo(k, pos), v4(0, 0, 1, 0));
        expect(Math.abs(form(k, fwdWorld, vertical))).toBeLessThan(1e-9);
      }
    });
});
