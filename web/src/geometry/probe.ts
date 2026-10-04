/**
 * GLSL-vs-TS geometry probe (§6 of the build prompt: "render a probe texture of test
 * cases and read it back"). Generates random rays/planes/balls for each κ, evaluates
 * both implementations and returns per-function max errors.
 */
import { FULLSCREEN_VS, FullscreenQuad, Program, TEX_W, dataTexture, renderTarget } from "../render/gl";
import { preprocess } from "../render/shaders";
import probeSrc from "../../shaders/geometry_probe.frag?raw";
import {
  type Kappa, v4, embedPoint, tangentialize, geodesic, geodesicDir, tangentToward, distance, planeRoot,
  ballInterval, planeExitAfter, planeEntryBefore, translationTo, apply, inverse, mul, logAtOrigin, dot4,
} from "./space";

export interface ProbeResult { name: string; kappa: Kappa; maxAbsErr: number; maxRelErr: number; n: number; mismatches: number }

function mulberry(seed: number) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

export function runGeometryProbe(gl: WebGL2RenderingContext, perKappa = 4096): ProbeResult[] {
  const R = mulberry(7);
  const randn = () => { const u = 1 - R(), v = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const kappas: Kappa[] = [-1, 0, 1];
  const N = perKappa * 3;
  const O = new Float32Array(N * 4), V = new Float32Array(N * 4), W = new Float32Array(N * 4), P = new Float32Array(N * 4), PR = new Float32Array(N * 4);
  const cases: { k: Kappa; o: Float64Array; v: Float64Array; w: Float64Array; p: Float64Array; t: number; r: number; tMin: number }[] = [];
  for (let ki = 0; ki < 3; ki++) {
    const k = kappas[ki];
    for (let i = 0; i < perKappa; i++) {
      const o = embedPoint(k, [randn() * 0.6, randn() * 0.6, randn() * 0.6]);
      const v = tangentialize(k, o, v4(0, randn(), randn(), randn()));
      const w = v4(randn() * 0.5, randn(), randn(), randn());
      const p = embedPoint(k, [randn() * 0.6, randn() * 0.6, randn() * 0.6]);
      const t = (R() - 0.5) * 4, r = 0.05 + R() * 1.2, tMin = R() * 3;
      const idx = cases.length;
      cases.push({ k, o, v, w, p, t, r, tMin });
      // store fp32 copies and use THOSE for the TS evaluation so both sides see identical inputs
      O.set(o, idx * 4); V.set(v, idx * 4); W.set(w, idx * 4); P.set(p, idx * 4); PR.set([t, r, tMin, k], idx * 4);
    }
  }
  const f64 = (a: Float32Array, i: number) => new Float64Array([a[i * 4], a[i * 4 + 1], a[i * 4 + 2], a[i * 4 + 3]]);
  const tex = (a: Float32Array, l: string) => dataTexture(gl, a, N, 4, gl.RGBA32F, gl.RGBA, gl.FLOAT, l);
  const tO = tex(O, "o"), tV = tex(V, "v"), tW = tex(W, "w"), tP = tex(P, "p"), tPR = tex(PR, "param");
  const rows = Math.ceil(N / TEX_W);
  const rt = renderTarget(gl, TEX_W, rows, gl.RGBA32F);
  const prog = new Program(gl, FULLSCREEN_VS, preprocess(probeSrc), "geometry_probe");
  const quad = new FullscreenQuad(gl);
  const out = new Float32Array(TEX_W * rows * 4);

  const runMode = (mode: number) => {
    gl.bindFramebuffer(gl.FRAMEBUFFER, rt.fbo);
    gl.viewport(0, 0, TEX_W, rows);
    prog.use();
    prog.tex("uO", 0, tO.tex); prog.tex("uV", 1, tV.tex); prog.tex("uW", 2, tW.tex); prog.tex("uP", 3, tP.tex); prog.tex("uParam", 4, tPR.tex);
    prog.u1i("uMode", mode); prog.u1i("uCount", N);
    quad.draw();
    gl.readPixels(0, 0, TEX_W, rows, gl.RGBA, gl.FLOAT, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out;
  };

  const results: ProbeResult[] = [];
  const compare = (name: string, mode: number, expected: (i: number) => number[] | null, tolAbs = 2e-4) => {
    const got = runMode(mode);
    for (const k of kappas) {
      let maxAbs = 0, maxRel = 0, n = 0, mism = 0;
      for (let i = 0; i < N; i++) {
        if (cases[i].k !== k) continue;
        const e = expected(i);
        if (!e) continue;
        n++;
        for (let c = 0; c < e.length; c++) {
          const g = got[i * 4 + c], x = e[c];
          if (Math.abs(x) >= 1e29) { if (Math.abs(g) < 1e29) mism++; continue; }
          const ae = Math.abs(g - x);
          maxAbs = Math.max(maxAbs, ae);
          maxRel = Math.max(maxRel, ae / Math.max(1, Math.abs(x)));
          if (ae > tolAbs * Math.max(1, Math.abs(x))) mism++;
        }
      }
      results.push({ name, kappa: k, maxAbsErr: maxAbs, maxRelErr: maxRel, n, mismatches: mism });
    }
  };
  const C = (i: number) => ({ k: cases[i].k, o: f64(O, i), v: f64(V, i), w: f64(W, i), p: f64(P, i), t: PR[i * 4], r: PR[i * 4 + 1], tMin: PR[i * 4 + 2] });

  compare("Eq.3 geodesic", 0, (i) => { const c = C(i); return Array.from(geodesic(c.k, c.o, c.v, c.t)); });
  compare("Eq.4 geodesicDir", 1, (i) => { const c = C(i); return Array.from(geodesicDir(c.k, c.o, c.v, c.t)); });
  compare("Eq.5 tangentToward", 2, (i) => { const c = C(i); return Array.from(tangentToward(c.k, c.o, c.p).u); }, 5e-4);
  compare("Eq.2 distance + Eq.6 planeRoot", 3, (i) => {
    const c = C(i); const A = dot4(c.w, c.o), B = dot4(c.w, c.v);
    const h = planeRoot(c.k, A, B);
    // near-degenerate roots (|B|≈|A| in H³, tiny denominators) are precision-limited; skip them
    if (c.k <= 0 && Math.abs(Math.abs(B) - Math.abs(A)) < 1e-2) return null;
    if (c.k === 0 && Math.abs(B) < 1e-2) return null;
    if (h.has && Math.abs(h.t) > 20) return null;
    return [distance(c.k, c.o, c.p), h.has ? h.t : 0, h.exit ? 1 : 0, h.has ? 1 : 0];
  }, 1e-3);
  compare("Eq.7 ballInterval + planeExitAfter", 4, (i) => {
    const c = C(i); const A = dot4(c.w, c.o), B = dot4(c.w, c.v);
    const b = ballInterval(c.k, c.o, c.v, c.p, c.r, c.tMin);
    const ea = planeExitAfter(c.k, A, B, c.tMin);
    if (c.k <= 0 && Math.abs(Math.abs(B) - Math.abs(A)) < 1e-2) return null;
    if (c.k === 0 && Math.abs(B) < 1e-2) return null;
    // tangent-grazing balls are precision-limited in fp32
    if (b.has && Math.abs(b.t2 - b.t1) < 2e-2) return null;
    if (!b.has) {
      const AA = c.k === 0 ? 0 : 0; void AA;
    }
    return [b.has ? 1 : 0, b.has ? b.t1 : 0, b.has ? b.t2 : 0, Number.isFinite(ea) ? ea : 1e30];
  }, 2e-3);
  compare("§3.6 embedPoint", 5, (i) => { const c = C(i); return Array.from(embedPoint(c.k, [c.p[1], c.p[2], c.p[3]])); });
  compare("§3.7 translationTo", 6, (i) => { const c = C(i); return Array.from(apply(translationTo(c.k, c.p), c.v)); });
  compare("§3.7 inverse∘translation = id", 7, (i) => { const c = C(i); const T = translationTo(c.k, c.p); return Array.from(apply(mul(inverse(c.k, T), T), c.v)); });
  compare("Eq.6 planeEntryBefore + logAtOrigin", 8, (i) => {
    const c = C(i); const A = dot4(c.w, c.o), B = dot4(c.w, c.v);
    const eb = planeEntryBefore(c.k, A, B, c.tMin + 10);
    if (c.k <= 0 && Math.abs(Math.abs(B) - Math.abs(A)) < 1e-2) return null;
    if (c.k === 0 && Math.abs(B) < 1e-2) return null;
    if (Number.isFinite(eb) && Math.abs(eb) > 20) return null;
    const l = logAtOrigin(c.k, c.p);
    return [Number.isFinite(eb) ? eb : -1e30, l[0], l[1], l[2]];
  }, 1e-3);
  return results;
}
