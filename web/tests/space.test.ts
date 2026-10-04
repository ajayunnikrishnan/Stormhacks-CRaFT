import { describe, expect, it } from "vitest";
import {
  type Kappa, type V4, ORIGIN, v4, form, cs, sn, distance, geodesic, geodesicDir, tangentToward,
  planeRoot, planeExitAfter, planeEntryBefore, planeValue, ballInterval, embedPoint, logAtOrigin, project,
  translationTo, translationByVector, rotation, inverse, isometryError, reorthonormalize, mul, apply,
  identity, advance, tangentialize, dot4,
} from "../src/geometry/space";

const KAPPAS: Kappa[] = [-1, 0, 1];

// deterministic PRNG (mulberry32)
function rng(seed: number) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const R = rng(42);
const randn = () => { const u = 1 - R(), v = R(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
const rand3 = (scale = 1) => [randn() * scale, randn() * scale, randn() * scale];

/** random point on the model and a unit tangent there */
function randomRay(k: Kappa, scale = 0.8): { o: V4; v: V4 } {
  const o = embedPoint(k, rand3(scale));
  const v = tangentialize(k, o, v4(0, ...rand3()));
  return { o, v };
}
function onModelError(k: Kappa, x: V4): number {
  return k === 0 ? Math.abs(x[0] - 1) : Math.abs(form(k, x, x) - k);
}
/** tangent-to-model condition: ⟨x,v⟩_κ = 0 for κ≠0; v0 = 0 for E³ (the form is degenerate there) */
function tangencyError(k: Kappa, x: V4, v: V4): number {
  return k === 0 ? Math.abs(v[0]) : Math.abs(form(k, x, v));
}
function rotationMatrix3(axis: number[], ang: number): number[] {
  const n = Math.hypot(...axis); const [x, y, z] = axis.map((a) => a / n);
  const c = Math.cos(ang), s = Math.sin(ang), C = 1 - c;
  // column-major
  return [c + x * x * C, y * x * C + z * s, z * x * C - y * s, x * y * C - z * s, c + y * y * C, z * y * C + x * s, x * z * C + y * s, y * z * C - x * s, c + z * z * C];
}
/** bisection on a scalar function over [a,b] with a sign change */
function bisect(f: (t: number) => number, a: number, b: number, iters = 200): number {
  let fa = f(a);
  for (let i = 0; i < iters; i++) { const m = 0.5 * (a + b); const fm = f(m); if ((fm < 0) === (fa < 0)) { a = m; fa = fm; } else b = m; }
  return 0.5 * (a + b);
}
/** all sign changes of f on [lo,hi] sampled on a grid, refined by bisection */
function numericRoots(f: (t: number) => number, lo: number, hi: number, n = 4000): number[] {
  const roots: number[] = [];
  let tPrev = lo, fPrev = f(lo);
  for (let i = 1; i <= n; i++) {
    const t = lo + ((hi - lo) * i) / n; const ft = f(t);
    if ((ft < 0) !== (fPrev < 0)) roots.push(bisect(f, tPrev, t));
    tPrev = t; fPrev = ft;
  }
  return roots;
}

describe("model constraints (Eqs. 3–4)", () => {
  for (const k of KAPPAS)
    it(`κ=${k}: γ stays on the model, γ' stays unit and tangent`, () => {
      for (let n = 0; n < 200; n++) {
        const { o, v } = randomRay(k);
        expect(onModelError(k, o)).toBeLessThan(1e-12);
        expect(Math.abs(form(k, v, v) - 1)).toBeLessThan(1e-12);
        expect(Math.abs(tangencyError(k, o, v))).toBeLessThan(1e-12);
        const t = (R() - 0.5) * 6;
        const g = geodesic(k, o, v, t), gd = geodesicDir(k, o, v, t);
        expect(onModelError(k, g)).toBeLessThan(1e-9);
        expect(Math.abs(form(k, gd, gd) - 1)).toBeLessThan(1e-9);
        expect(Math.abs(tangencyError(k, g, gd))).toBeLessThan(1e-9);
        // arc length: d(o, γ(t)) = |t| (S³: for |t| ≤ π)
        if (k <= 0 || Math.abs(t) <= Math.PI) expect(Math.abs(distance(k, o, g) - Math.abs(t))).toBeLessThan(1e-9);
      }
    });
  it("distance matches Eq. (2) cs(d) = κ⟨x,y⟩ and is accurate for tiny d", () => {
    for (const k of [-1, 1] as Kappa[])
      for (let n = 0; n < 100; n++) {
        const x = embedPoint(k, rand3()), y = embedPoint(k, rand3());
        const d = distance(k, x, y);
        expect(Math.abs(cs(k, d) - k * form(k, x, y))).toBeLessThan(1e-9);
        const { o, v } = randomRay(k);
        const tiny = 1e-7;
        expect(Math.abs(distance(k, o, geodesic(k, o, v, tiny)) - tiny) / tiny).toBeLessThan(1e-6);
      }
  });
});

describe("Eq. (5) tangent toward a point round-trips", () => {
  for (const k of KAPPAS)
    it(`κ=${k}`, () => {
      for (let n = 0; n < 200; n++) {
        const { o, v } = randomRay(k);
        const d = 0.01 + R() * (k > 0 ? 2.5 : 3);
        const y = geodesic(k, o, v, d);
        const { u, d: d2 } = tangentToward(k, o, y);
        expect(Math.abs(d2 - d)).toBeLessThan(1e-9);
        for (let i = 0; i < 4; i++) expect(Math.abs(u[i] - v[i])).toBeLessThan(1e-8);
        expect(Math.abs(form(k, u, u) - 1)).toBeLessThan(1e-9);
        expect(Math.abs(tangencyError(k, o, u))).toBeLessThan(1e-9);
      }
    });
});

describe("Eq. (6) plane crossing vs numeric roots", () => {
  for (const k of KAPPAS)
    it(`κ=${k}: closed-form root, exit/entry sign`, () => {
      let tested = 0;
      for (let n = 0; n < 400; n++) {
        const { o, v } = randomRay(k);
        // random plane covector; for curved spaces w = J n with random n
        const w = v4(...rand3(), randn());
        if (k === 0) w[0] = randn() * 0.5; // E³ offset term
        const A = dot4(w, o), B = dot4(w, v);
        const f = (t: number) => planeValue(k, A, B, t);
        const lo = k > 0 ? 0 : -8, hi = k > 0 ? 2 * Math.PI : 8;
        const roots = numericRoots(f, lo, hi);
        const h = planeRoot(k, A, B);
        if (k > 0) {
          expect(roots.length).toBe(2);
          // our exit root is one of them and has positive slope
          const near = roots.filter((r) => Math.abs(r - h.t) < 1e-6);
          expect(near.length).toBe(1);
          const slope = (f(h.t + 1e-6) - f(h.t - 1e-6)) / 2e-6;
          expect(slope).toBeGreaterThan(0);
          // and the entry root has negative slope
          const te = planeEntryBefore(k, A, B, 2 * Math.PI);
          expect(Math.abs(f(te))).toBeLessThan(1e-9);
          expect((f(te + 1e-6) - f(te - 1e-6)) / 2e-6).toBeLessThan(0);
          // periodicity helpers
          const tm = R() * 20;
          const ex = planeExitAfter(k, A, B, tm);
          expect(ex).toBeGreaterThanOrEqual(tm - 1e-12);
          expect(ex - tm).toBeLessThan(2 * Math.PI + 1e-9);
          expect(Math.abs(f(ex))).toBeLessThan(1e-8);
        } else {
          const inRange = roots.filter((r) => Math.abs(r) < 7.9);
          if (!h.has) { expect(inRange.length).toBe(0); continue; }
          if (Math.abs(h.t) > 7.9) continue;
          expect(inRange.length).toBe(1);
          expect(Math.abs(inRange[0] - h.t)).toBeLessThan(1e-7);
          const slope = (f(h.t + 1e-6) - f(h.t - 1e-6)) / 2e-6;
          expect(slope > 0).toBe(h.exit);
          tested++;
        }
      }
      if (k <= 0) expect(tested).toBeGreaterThan(100);
    });
});

describe("Eq. (7) ball interval vs numeric roots", () => {
  for (const k of KAPPAS)
    it(`κ=${k}`, () => {
      let hits = 0;
      for (let n = 0; n < 400; n++) {
        const { o, v } = randomRay(k, 0.6);
        const p = embedPoint(k, rand3(0.6));
        const r = 0.05 + R() * (k > 0 ? 1.2 : 1.5);
        const inside = (t: number) => distance(k, p, geodesic(k, o, v, t)) - r; // ≤ 0 inside
        const b = ballInterval(k, o, v, p, r, k > 0 ? -Math.PI : -Infinity);
        const lo = k > 0 ? -Math.PI : -10, hi = k > 0 ? Math.PI : 10;
        const roots = numericRoots(inside, lo, hi, 6000);
        if (!b.has) { expect(roots.length).toBe(0); continue; }
        hits++;
        expect(b.t1).toBeLessThanOrEqual(b.t2);
        expect(Math.abs(inside(b.t1))).toBeLessThan(1e-7);
        expect(Math.abs(inside(b.t2))).toBeLessThan(1e-7);
        expect(inside(0.5 * (b.t1 + b.t2))).toBeLessThan(0);
        if (k <= 0) { expect(roots.length).toBe(2); expect(Math.abs(roots[0] - b.t1)).toBeLessThan(1e-6); expect(Math.abs(roots[1] - b.t2)).toBeLessThan(1e-6); }
      }
      expect(hits).toBeGreaterThan(50);
    });
  it("S³: interval selection respects tMin and is 2π-periodic", () => {
    const k: Kappa = 1;
    for (let n = 0; n < 100; n++) {
      const { o, v } = randomRay(k, 0.6);
      const p = embedPoint(k, rand3(0.6));
      const b0 = ballInterval(k, o, v, p, 0.5, 0);
      if (!b0.has) continue;
      expect(b0.t2).toBeGreaterThanOrEqual(0);
      expect(b0.t2 - 2 * Math.PI).toBeLessThan(0);
      const b1 = ballInterval(k, o, v, p, 0.5, b0.t2 + 1e-6);
      expect(Math.abs(b1.t1 - b0.t1 - 2 * Math.PI)).toBeLessThan(1e-9);
    }
  });
});

describe("§3.6 embedding", () => {
  for (const k of KAPPAS)
    it(`κ=${k}: embed/log round trip, distance from origin = |x̄|`, () => {
      for (let n = 0; n < 200; n++) {
        const x = rand3(0.7);
        const p = embedPoint(k, x);
        expect(onModelError(k, p)).toBeLessThan(1e-12);
        expect(Math.abs(distance(k, ORIGIN, p) - Math.hypot(...x))).toBeLessThan(1e-9);
        const back = logAtOrigin(k, p);
        for (let i = 0; i < 3; i++) expect(Math.abs(back[i] - x[i])).toBeLessThan(1e-9);
      }
    });
});

describe("§3.7 isometries", () => {
  for (const k of KAPPAS)
    it(`κ=${k}: translation maps o→p, satisfies MᵀJM = J, inverse works, preserves distances`, () => {
      for (let n = 0; n < 200; n++) {
        const p = embedPoint(k, rand3(0.8));
        const T = translationTo(k, p);
        expect(isometryError(k, T)).toBeLessThan(1e-12);
        const img = apply(T, ORIGIN);
        for (let i = 0; i < 4; i++) expect(Math.abs(img[i] - p[i])).toBeLessThan(1e-12);
        const Ti = inverse(k, T);
        const I = mul(Ti, T);
        const Id = identity();
        for (let i = 0; i < 16; i++) expect(Math.abs(I[i] - Id[i])).toBeLessThan(1e-12);
        // T_p⁻¹ = T_{-p̄}
        const Tneg = translationTo(k, embedPoint(k, logAtOrigin(k, p).map((c) => -c)));
        for (let i = 0; i < 16; i++) expect(Math.abs(Tneg[i] - Ti[i])).toBeLessThan(1e-12);
        const x = embedPoint(k, rand3()), y = embedPoint(k, rand3());
        const Rm = rotation(rotationMatrix3(rand3(), R() * 6));
        const M = mul(Rm, T);
        expect(isometryError(k, M)).toBeLessThan(1e-12);
        expect(Math.abs(distance(k, apply(M, x), apply(M, y)) - distance(k, x, y))).toBeLessThan(1e-9);
      }
    });
  it("translationByVector moves the origin by exactly |v| along v", () => {
    for (const k of KAPPAS) {
      const v = [0.3, -0.2, 0.5];
      const T = translationByVector(k, v);
      const p = apply(T, ORIGIN);
      expect(Math.abs(distance(k, ORIGIN, p) - Math.hypot(...v))).toBeLessThan(1e-12);
      const { u } = tangentToward(k, ORIGIN, p);
      const n = Math.hypot(...v);
      for (let i = 0; i < 3; i++) expect(Math.abs(u[i + 1] - v[i] / n)).toBeLessThan(1e-12);
    }
  });
  it("transport: advancing a ray and re-advancing equals advancing once (Eqs. 3–4 compose)", () => {
    for (const k of KAPPAS)
      for (let n = 0; n < 100; n++) {
        const ray = randomRay(k);
        const a = R() * 2, b = R() * 2;
        const r1 = advance(k, advance(k, ray, a), b);
        const r2 = advance(k, ray, a + b);
        for (let i = 0; i < 4; i++) { expect(Math.abs(r1.o[i] - r2.o[i])).toBeLessThan(1e-9); expect(Math.abs(r1.v[i] - r2.v[i])).toBeLessThan(1e-9); }
      }
  });
});

describe("drift test: 10⁶ random moves with re-orthonormalisation (§3.7)", () => {
  // The walk is kept within ~2 units of the world origin: in the game W always maps the
  // world into a bounded region (the fundamental domain re-centres the camera on every
  // face crossing). An UNBOUNDED walk in H³ wanders ~12 units in 10⁶ steps, where the
  // hyperboloid coordinates reach ~1e5 and fp64 cancellation in ⟨x,x⟩ alone exceeds 1e-5
  // — a property of the model, not of the re-orthonormalisation (see docs/WRITEUP.md).
  for (const k of KAPPAS)
    it(`κ=${k}: ‖WᵀJW − J‖ < 1e-5 (bounded walk)`, () => {
      let W = identity();
      const tmp = new Float64Array(16);
      const N = 1_000_000;
      for (let i = 0; i < N; i++) {
        let step: Float64Array;
        if (i % 2 === 0) {
          const d = [randn() * 0.01, randn() * 0.01, randn() * 0.01];
          // camera position in world coords is W⁻¹ o; its log-map tells us how far we are
          const camWorld = apply(inverse(k, W), ORIGIN);
          const off = logAtOrigin(k, camWorld);
          if (Math.hypot(off[0], off[1], off[2]) > 2) {
            // step toward the world origin instead (in camera coordinates: toward W·o)
            const toOrigin = tangentToward(k, ORIGIN, apply(W, ORIGIN)).u;
            for (let j = 0; j < 3; j++) d[j] = -0.01 * toOrigin[j + 1]; // W ← T(−δ·dir)·W moves the camera along dir
          }
          step = translationByVector(k, d, tmp);
        } else step = rotation(rotationMatrix3(rand3(), randn() * 0.02), tmp);
        mul(step, W, W);
        reorthonormalize(k, W);
      }
      expect(isometryError(k, W)).toBeLessThan(1e-5);
      // and the camera stays a valid point (first column on the model, x0>0 for H³)
      const c0 = new Float64Array(W.buffer, 0, 4);
      expect(onModelError(k, c0)).toBeLessThan(1e-9);
      if (k < 0) expect(c0[0]).toBeGreaterThan(0);
    }, 60_000);
});

describe("flat limit: curved results → Euclidean as the scene scale s → 0 (§3.1)", () => {
  it("distances, plane crossings and ball intervals converge with O(s²) error", () => {
    const a = [0.3, 0.1, -0.2], b = [-0.4, 0.25, 0.6], p = [0.1, 0.5, 0.2], r = 0.35;
    const dir = [0.6, -0.3, 0.74]; const dn = Math.hypot(...dir); const d3 = dir.map((c) => c / dn);
    const flatD = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    for (const k of [-1, 1] as Kappa[]) {
      let prevErr = Infinity;
      for (const s of [0.4, 0.2, 0.1, 0.05, 0.025]) {
        const A = embedPoint(k, a.map((c) => c * s)), B = embedPoint(k, b.map((c) => c * s));
        const errD = Math.abs(distance(k, A, B) / s - flatD);
        expect(errD).toBeLessThan(prevErr + 1e-15);
        prevErr = errD;
        // ball interval from a scaled ray: compare with the Euclidean quadratic
        const o = A, v = tangentialize(k, o, v4(0, ...d3));
        const curved = ballInterval(k, o, v, embedPoint(k, p.map((c) => c * s)), r * s);
        const flat = ballInterval(0, v4(1, ...a), v4(0, ...d3), v4(1, ...p), r);
        expect(curved.has).toBe(flat.has);
        if (flat.has) { expect(Math.abs(curved.t1 / s - flat.t1)).toBeLessThan(4 * s * s); expect(Math.abs(curved.t2 / s - flat.t2)).toBeLessThan(4 * s * s); }
      }
      expect(prevErr).toBeLessThan(1e-3);
    }
  });
});

describe("project", () => {
  it("re-projects a perturbed point back to the model", () => {
    for (const k of KAPPAS) {
      const x = embedPoint(k, [0.3, 0.2, 0.1]);
      for (let i = 0; i < 4; i++) x[i] *= 1 + 1e-4 * (i + 1);
      project(k, x);
      expect(onModelError(k, x)).toBeLessThan(1e-12);
    }
  });
});
