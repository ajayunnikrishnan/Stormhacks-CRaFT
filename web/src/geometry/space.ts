/**
 * Unified constant-curvature geometry for H³ (κ=−1), E³ (κ=0), S³ (κ=+1).
 *
 * Mirrored line for line by shaders/geometry.glsl — keep both in sync, same
 * equation numbers. Equation numbers refer to docs/WRITEUP.md (= build prompt §3).
 *
 * Ambient ℝ⁴ coordinates x = (x0, x1, x2, x3).
 *   H³: ⟨x,x⟩ = −1, x0 > 0          S³: ⟨x,x⟩ = +1          E³: x0 = 1, x̄ ∈ ℝ³
 * Tangent vectors have ⟨x,v⟩ = 0 (E³: v0 = 0). Points/vectors are Float64Array(4);
 * matrices are column-major Float64Array(16) (m[col*4 + row]) like GLSL mat4.
 *
 * Planes are stored as covectors w: plane = { x : w·x = 0 } with the ORDINARY dot
 * product. For κ≠0 the plane with κ-normal n is w = J n (J = diag(κ,1,1,1)); for
 * E³ the plane n̄·x̄ = c is w = (−c, n̄). This makes Eq. (6) one formula for all κ.
 */

export type Kappa = -1 | 0 | 1;
export type V4 = Float64Array; // length 4
export type M4 = Float64Array; // length 16, column-major

export const EPS = 1e-12;

export const v4 = (a = 0, b = 0, c = 0, d = 0): V4 => new Float64Array([a, b, c, d]);
export const ORIGIN: V4 = v4(1, 0, 0, 0); // o = (1,0,0,0) for every κ

// ---------------------------------------------------------------------------
// Eq. (1): bilinear form ⟨x,y⟩_κ = κ x0 y0 + x1 y1 + x2 y2 + x3 y3
// ---------------------------------------------------------------------------
export function form(k: Kappa, x: V4, y: V4): number {
  return k * x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}
/** ordinary ℝ⁴ dot product (used with plane covectors). */
export function dot4(x: V4, y: V4): number {
  return x[0] * y[0] + x[1] * y[1] + x[2] * y[2] + x[3] * y[3];
}

// ---------------------------------------------------------------------------
// Generalised trigonometry: sn_κ = sin/sinh, cs_κ = cos/cosh; E³: sn=t, cs=1.
// ---------------------------------------------------------------------------
export function cs(k: Kappa, t: number): number {
  return k > 0 ? Math.cos(t) : k < 0 ? Math.cosh(t) : 1;
}
export function sn(k: Kappa, t: number): number {
  return k > 0 ? Math.sin(t) : k < 0 ? Math.sinh(t) : t;
}
export function tn(k: Kappa, t: number): number {
  return sn(k, t) / cs(k, t);
}
/** inverse of cs on the principal branch (argument clamped to the domain). */
export function acs(k: Kappa, c: number): number {
  return k > 0 ? Math.acos(Math.max(-1, Math.min(1, c))) : k < 0 ? Math.acosh(Math.max(1, c)) : 0;
}

// ---------------------------------------------------------------------------
// Eq. (2): cs_κ(d(x,y)) = κ⟨x,y⟩_κ. Implemented through the chord length
//   c² = ⟨x−y,x−y⟩_κ = 2κ − 2⟨x,y⟩ = 2κ(1 − cs_κ d) = 4 sinh²(d/2) (H³) / 4 sin²(d/2) (S³)
//   ⇒  H³: d = 2 asinh(c/2),  S³: d = 2 asin(c/2)
// which is accurate for small d where acosh(1+ε) is not. (c² ≥ 0 for both signs of κ.)
// ---------------------------------------------------------------------------
export function distance(k: Kappa, x: V4, y: V4): number {
  const d0 = x[0] - y[0], d1 = x[1] - y[1], d2 = x[2] - y[2], d3 = x[3] - y[3];
  if (k === 0) return Math.sqrt(d1 * d1 + d2 * d2 + d3 * d3);
  const c2 = k * d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3; // = ⟨x−y,x−y⟩_κ ≥ 0
  const half = Math.sqrt(Math.max(0, c2)) / 2;
  return k > 0 ? 2 * Math.asin(Math.min(1, half)) : 2 * Math.asinh(half);
}

// ---------------------------------------------------------------------------
// Eq. (3): γ(t) = cs_κ(t)·o' + sn_κ(t)·v     (unit-speed geodesic, t = arc length)
// Eq. (4): γ'(t) = −κ·sn_κ(t)·o' + cs_κ(t)·v (parallel-transported direction)
// ---------------------------------------------------------------------------
export function geodesic(k: Kappa, o: V4, v: V4, t: number, out: V4 = v4()): V4 {
  const c = cs(k, t), s = sn(k, t);
  for (let i = 0; i < 4; i++) out[i] = c * o[i] + s * v[i];
  return out;
}
export function geodesicDir(k: Kappa, o: V4, v: V4, t: number, out: V4 = v4()): V4 {
  const c = cs(k, t), s = sn(k, t);
  for (let i = 0; i < 4; i++) out[i] = -k * s * o[i] + c * v[i];
  return out;
}

// ---------------------------------------------------------------------------
// Eq. (5): unit tangent at x pointing toward y:  u = (y − cs_κ(d)·x) / sn_κ(d)
// ---------------------------------------------------------------------------
export function tangentToward(k: Kappa, x: V4, y: V4, out: V4 = v4()): { u: V4; d: number } {
  const d = distance(k, x, y);
  const c = cs(k, d), s = sn(k, d);
  if (s < EPS) { out.fill(0); return { u: out, d }; }
  for (let i = 0; i < 4; i++) out[i] = (y[i] - c * x[i]) / s;
  return { u: out, d };
}

// ---------------------------------------------------------------------------
// Eq. (6): ray vs linear plane {w·x = 0}. With A = w·o', B = w·v, along the ray
//   f(t) = A·cs_κ(t) + B·sn_κ(t) = 0.
// The "cell side" is w·x ≤ 0. A root is an EXIT if f'(t) > 0 there, an ENTRY if < 0.
//   E³: t = −A/B,            exit ⇔ B > 0
//   H³: with E = e^t: (A+B)E² + (A−B) = 0 ⇒ t = ½ ln((B−A)/(A+B)), exists ⇔ |B| > |A|,
//       f'(t) = cosh t·(B² − A²)/B ⇒ exit ⇔ B > 0
//   S³: f(t) = R cos(t − φ), R = √(A²+B²), φ = atan2(B, A); roots t = φ ± π/2 (+2πk);
//       f' = −R sin(t−φ): t = φ − π/2 is an EXIT, t = φ + π/2 an ENTRY.
// ---------------------------------------------------------------------------
export interface PlaneHit { has: boolean; t: number; exit: boolean }

/** Single-root form (E³, H³); for S³ returns the exit root in [0, 2π). */
export function planeRoot(k: Kappa, A: number, B: number): PlaneHit {
  if (k === 0) {
    if (B === 0) return { has: false, t: 0, exit: false };
    return { has: true, t: -A / B, exit: B > 0 };
  }
  if (k < 0) {
    if (Math.abs(B) <= Math.abs(A)) return { has: false, t: 0, exit: false };
    return { has: true, t: 0.5 * Math.log((B - A) / (A + B)), exit: B > 0 };
  }
  const R = Math.hypot(A, B);
  if (R < EPS) return { has: false, t: 0, exit: false };
  const phi = Math.atan2(B, A);
  let t = phi - Math.PI / 2; // exit root
  t -= 2 * Math.PI * Math.floor(t / (2 * Math.PI)); // wrap to [0, 2π)
  return { has: true, t, exit: true };
}

/** First EXIT root with t ≥ tMin, or +∞ (S³: periodic in 2π; E³/H³: single root). */
export function planeExitAfter(k: Kappa, A: number, B: number, tMin: number): number {
  if (k > 0) {
    const R = Math.hypot(A, B);
    if (R < EPS) return Infinity;
    const base = Math.atan2(B, A) - Math.PI / 2;
    const TWO_PI = 2 * Math.PI;
    return base + TWO_PI * Math.ceil((tMin - base) / TWO_PI);
  }
  const h = planeRoot(k, A, B);
  return h.has && h.exit && h.t >= tMin ? h.t : Infinity;
}

/** Last ENTRY root with t ≤ tMax, or −∞. */
export function planeEntryBefore(k: Kappa, A: number, B: number, tMax: number): number {
  if (k > 0) {
    const R = Math.hypot(A, B);
    if (R < EPS) return -Infinity;
    const base = Math.atan2(B, A) + Math.PI / 2;
    const TWO_PI = 2 * Math.PI;
    return base + TWO_PI * Math.floor((tMax - base) / TWO_PI);
  }
  const h = planeRoot(k, A, B);
  return h.has && !h.exit && h.t <= tMax ? h.t : -Infinity;
}

/** Fused exit/entry solve (mirror of planeExitEntryK): [first exit ≥ tRef, last entry ≤ tRef]. */
export function planeExitEntry(k: Kappa, A: number, B: number, tRef: number): [number, number] {
  if (k > 0) {
    if (A * A + B * B < EPS * EPS) return [Infinity, -Infinity];
    const phi = Math.atan2(B, A), TWO_PI = 2 * Math.PI;
    const bx = phi - Math.PI / 2, bn = phi + Math.PI / 2;
    return [bx + TWO_PI * Math.ceil((tRef - bx) / TWO_PI), bn + TWO_PI * Math.floor((tRef - bn) / TWO_PI)];
  }
  let t: number;
  if (k === 0) { if (B === 0) return [Infinity, -Infinity]; t = -A / B; }
  else { if (Math.abs(B) <= Math.abs(A)) return [Infinity, -Infinity]; t = 0.5 * Math.log((B - A) / (A + B)); }
  if (B > 0) return [t >= tRef ? t : Infinity, -Infinity];
  return [Infinity, t <= tRef ? t : -Infinity];
}

/** Signed plane value along the ray at t (for tests / inside checks). */
export function planeValue(k: Kappa, A: number, B: number, t: number): number {
  return A * cs(k, t) + B * sn(k, t);
}

// ---------------------------------------------------------------------------
// Eq. (7): ray vs geodesic ball of radius r about site p. With A = ⟨o',p⟩, B = ⟨v,p⟩:
//   inside ⇔ d(γ(t),p) ≤ r ⇔ A·cs_κ(t) + B·sn_κ(t) ≥ κ·cs_κ(r)      (both signs of κ)
//   S³: R cos(t−φ) ≥ cos r  ⇒ t ∈ [φ−α, φ+α] + 2πk, α = acos(cos r / R), needs R ≥ cos r
//   H³: with E = e^t: (A+B)E² + 2cosh(r)E + (A−B) ≥ 0, leading coeff < 0 ⇒ between roots
//       E± = (−cosh r ± √(cosh²r − (A²−B²))) / (A+B), needs A²−B² ≤ cosh²r
//   E³: |ō + t v̄ − p̄|² ≤ r² (ordinary quadratic; A,B unused)
// CONDITIONING (same equations, rearranged for fp32 at small scale s): A = κ + A' with
//   A' = −½⟨o'−p, o'−p⟩_κ  (a small, accurately computed chord term), so that
//   H³: disc = sinh²r + 2A' − A'² + B²,  E = 1 + δ,  δ = (−2sinh²(r/2) − A' − B ± √disc)/(A+B)
//   S³: R − cos r = (2A' + A'² + B²)/(R+1) + 2sin²(r/2),  α = 2 asin(√(ε/2)), ε = (R − cos r)/R
// Returns the interval [t1,t2]; for S³ the one with t2 ≥ tMin closest to tMin.
// ---------------------------------------------------------------------------
export interface BallHit { has: boolean; t1: number; t2: number }

/** log(1+x) accurate for small x (GLSL mirror uses the same series). */
export function log1p(x: number): number {
  return Math.abs(x) < 1e-4 ? x - 0.5 * x * x + x * x * x / 3 : Math.log(1 + x);
}

export function ballInterval(k: Kappa, o: V4, v: V4, p: V4, r: number, tMin = 0): BallHit {
  if (k === 0) {
    const oc1 = o[1] - p[1], oc2 = o[2] - p[2], oc3 = o[3] - p[3];
    const qb = 2 * (oc1 * v[1] + oc2 * v[2] + oc3 * v[3]);
    const qc = oc1 * oc1 + oc2 * oc2 + oc3 * oc3 - r * r;
    const disc = qb * qb - 4 * qc;
    if (disc < 0) return { has: false, t1: 0, t2: 0 };
    const s = Math.sqrt(disc);
    return { has: true, t1: (-qb - s) / 2, t2: (-qb + s) / 2 };
  }
  const d0 = o[0] - p[0], d1 = o[1] - p[1], d2 = o[2] - p[2], d3 = o[3] - p[3];
  const Ap = -0.5 * (k * d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3); // A' = A − κ
  const B = form(k, v, p);
  if (k < 0) {
    const sr = Math.sinh(r), sh = Math.sinh(r / 2);
    const disc = sr * sr + 2 * Ap - Ap * Ap + B * B;
    if (disc < 0) return { has: false, t1: 0, t2: 0 };
    const sq = Math.sqrt(disc);
    const den = -1 + Ap + B; // A + B < 0
    const num0 = -2 * sh * sh - Ap - B;
    const dPlus = (num0 + sq) / den, dMinus = (num0 - sq) / den; // E = 1 + δ
    const ta = log1p(dPlus), tb = log1p(dMinus);
    return { has: true, t1: Math.min(ta, tb), t2: Math.max(ta, tb) };
  }
  const A = 1 + Ap;
  const R = Math.hypot(A, B);
  const sh = Math.sin(r / 2);
  const RmC = (2 * Ap + Ap * Ap + B * B) / (R + 1) + 2 * sh * sh; // R − cos r
  if (RmC < 0) return { has: false, t1: 0, t2: 0 };
  const eps = RmC / R; // 1 − cos r / R
  const alpha = 2 * Math.asin(Math.min(1, Math.sqrt(eps / 2)));
  const phi = Math.atan2(B, A);
  const TWO_PI = 2 * Math.PI;
  const kk = Math.ceil((tMin - (phi + alpha)) / TWO_PI);
  return { has: true, t1: phi - alpha + TWO_PI * kk, t2: phi + alpha + TWO_PI * kk };
}

// §3.6: embed a Euclidean point x̄ (already scaled by s = √|k|) at geodesic
// distance |x̄| from the origin along direction x̂:  p = cs(|x̄|)·o + sn(|x̄|)·(0, x̂)
// ---------------------------------------------------------------------------
export function embedPoint(k: Kappa, x: ArrayLike<number>, out: V4 = v4()): V4 {
  const n = Math.hypot(x[0], x[1], x[2]);
  if (k === 0) { out[0] = 1; out[1] = x[0]; out[2] = x[1]; out[3] = x[2]; return out; }
  if (n < EPS) { out[0] = 1; out[1] = out[2] = out[3] = 0; return out; }
  const s = sn(k, n) / n;
  out[0] = cs(k, n); out[1] = s * x[0]; out[2] = s * x[1]; out[3] = s * x[2];
  return out;
}

/** Inverse of embedPoint (log map at the origin): returns x̄ with |x̄| = d(o,p). */
export function logAtOrigin(k: Kappa, p: V4, out = new Float64Array(3)): Float64Array {
  if (k === 0) { out[0] = p[1]; out[1] = p[2]; out[2] = p[3]; return out; }
  const d = distance(k, ORIGIN, p);
  const s = sn(k, d);
  const f = s < EPS ? 1 : d / s;
  out[0] = f * p[1]; out[1] = f * p[2]; out[2] = f * p[3];
  return out;
}

/** Re-project a drifted point onto the model (⟨x,x⟩ = κ; E³: x0 = 1). */
export function project(k: Kappa, x: V4): V4 {
  if (k === 0) { x[0] = 1; return x; }
  const n = Math.sqrt(Math.max(EPS, k * form(k, x, x)));
  for (let i = 0; i < 4; i++) x[i] /= n;
  return x;
}

// ---------------------------------------------------------------------------
// §3.7: isometries as 4×4 matrices with MᵀJM = J, J = diag(κ,1,1,1).
// ---------------------------------------------------------------------------
export const m4 = (): M4 => new Float64Array(16);
export function identity(out: M4 = m4()): M4 {
  out.fill(0);
  out[0] = out[5] = out[10] = out[15] = 1;
  return out;
}
export function mul(a: M4, b: M4, out: M4 = m4()): M4 {
  const r = out === a || out === b ? m4() : out;
  for (let c = 0; c < 4; c++)
    for (let rr = 0; rr < 4; rr++) {
      let s = 0;
      for (let i = 0; i < 4; i++) s += a[i * 4 + rr] * b[c * 4 + i];
      r[c * 4 + rr] = s;
    }
  if (r !== out) out.set(r);
  return out;
}
export function apply(m: M4, x: V4, out: V4 = v4()): V4 {
  const r = out === x ? v4() : out;
  for (let rr = 0; rr < 4; rr++) r[rr] = m[rr] * x[0] + m[4 + rr] * x[1] + m[8 + rr] * x[2] + m[12 + rr] * x[3];
  if (r !== out) out.set(r);
  return out;
}

/**
 * Translation isometry T_p mapping the origin o to p = (c, s·û), c = cs(d), s = sn(d):
 *   T = [[ c, −κ s ûᵀ ], [ s û, I + (c−1) û ûᵀ ]]      (column-major below)
 * κ=+1: rotation in the (e0, û) plane; κ=−1: boost; κ=0: Euclidean translation by s û.
 * Its inverse is the translation to the antipodal direction (−û) with the same d.
 */
export function translationTo(k: Kappa, p: V4, out: M4 = m4()): M4 {
  const c = k === 0 ? 1 : p[0];
  const s = Math.hypot(p[1], p[2], p[3]);
  const u1 = s < EPS ? 0 : p[1] / s, u2 = s < EPS ? 0 : p[2] / s, u3 = s < EPS ? 0 : p[3] / s;
  const u = [u1, u2, u3];
  // column 0: image of e0 = (c, s û)
  out[0] = c; out[1] = s * u1; out[2] = s * u2; out[3] = s * u3;
  // columns 1..3: image of e_i = (−κ s u_i, e_i + (c−1) u_i û)
  for (let i = 0; i < 3; i++) {
    out[4 * (i + 1)] = -k * s * u[i];
    for (let j = 0; j < 3; j++) out[4 * (i + 1) + 1 + j] = (i === j ? 1 : 0) + (c - 1) * u[i] * u[j];
  }
  return out;
}

/** Translation by a tangent vector at the origin, (0, v̄) with |v̄| = distance. */
export function translationByVector(k: Kappa, v: ArrayLike<number>, out: M4 = m4()): M4 {
  return translationTo(k, embedPoint(k, v), out);
}

/** Pure rotation about the origin: block-diagonal diag(1, R3), R3 column-major 3×3. */
export function rotation(R3: ArrayLike<number>, out: M4 = m4()): M4 {
  identity(out);
  for (let c = 0; c < 3; c++) for (let r = 0; r < 3; r++) out[4 * (c + 1) + 1 + r] = R3[3 * c + r];
  return out;
}

/** Inverse of an isometry. κ≠0: M⁻¹ = J Mᵀ J; κ=0: block inverse [[1,0],[−Rᵀt, Rᵀ]]. */
export function inverse(k: Kappa, m: M4, out: M4 = m4()): M4 {
  const r = out === m ? m4() : out;
  if (k === 0) {
    identity(r);
    for (let c = 0; c < 3; c++) for (let rr = 0; rr < 3; rr++) r[4 * (c + 1) + 1 + rr] = m[4 * (rr + 1) + 1 + c];
    for (let rr = 0; rr < 3; rr++) {
      let s = 0;
      for (let c = 0; c < 3; c++) s -= m[4 * (rr + 1) + 1 + c] * m[1 + c]; // −Rᵀ t
      r[1 + rr] = s;
    }
  } else {
    // (J Mᵀ J)[row][col] = J_row · M[col][row] · J_col
    for (let c = 0; c < 4; c++)
      for (let rr = 0; rr < 4; rr++) {
        const jr = rr === 0 ? k : 1, jc = c === 0 ? k : 1;
        r[c * 4 + rr] = jr * m[rr * 4 + c] * jc;
      }
  }
  if (r !== out) out.set(r);
  return out;
}

/**
 * ‖MᵀJM − J‖_max — zero for an exact isometry. For E³ the form is degenerate
 * (J = diag(0,1,1,1) says nothing about the translation column), so there we
 * additionally require the top row to be (1,0,0,0).
 */
export function isometryError(k: Kappa, m: M4): number {
  let err = 0;
  const col = (i: number): V4 => new Float64Array(m.buffer, m.byteOffset + 32 * i, 4);
  if (k === 0) err = Math.max(Math.abs(m[0] - 1), Math.abs(m[4]), Math.abs(m[8]), Math.abs(m[12]));
  const first = k === 0 ? 1 : 0; // E³: the translation column is unconstrained by the degenerate form
  for (let i = first; i < 4; i++)
    for (let j = first; j < 4; j++) {
      const target = i === j ? (i === 0 ? k : 1) : 0;
      err = Math.max(err, Math.abs(form(k, col(i), col(j)) - target));
    }
  return err;
}

/**
 * Gram–Schmidt with respect to J on the columns (§3.7 drift control):
 * col0 ← col0/√(κ⟨col0,col0⟩) (so ⟨col0,col0⟩ = κ), then for i=1..3
 * col_i ← col_i − κ⟨col_i,col0⟩col0 − Σ_{0<j<i}⟨col_i,col_j⟩col_j, normalised.
 * E³: the top row is pinned to (1,0,0,0) and the rotation block is orthonormalised.
 */
export function reorthonormalize(k: Kappa, m: M4): M4 {
  const col = (i: number): V4 => new Float64Array(m.buffer, m.byteOffset + 32 * i, 4);
  if (k === 0) {
    m[0] = 1; m[4] = 0; m[8] = 0; m[12] = 0;
    for (let i = 1; i < 4; i++) {
      const ci = col(i);
      for (let j = 1; j < i; j++) {
        const cj = col(j);
        const d = ci[1] * cj[1] + ci[2] * cj[2] + ci[3] * cj[3];
        for (let r = 1; r < 4; r++) ci[r] -= d * cj[r];
      }
      const n = Math.hypot(ci[1], ci[2], ci[3]);
      for (let r = 1; r < 4; r++) ci[r] /= n;
    }
    return m;
  }
  const c0 = col(0);
  const n0 = Math.sqrt(k * form(k, c0, c0));
  for (let r = 0; r < 4; r++) c0[r] /= n0;
  for (let i = 1; i < 4; i++) {
    const ci = col(i);
    const d0 = k * form(k, ci, c0);
    for (let r = 0; r < 4; r++) ci[r] -= d0 * c0[r];
    for (let j = 1; j < i; j++) {
      const cj = col(j);
      const d = form(k, ci, cj);
      for (let r = 0; r < 4; r++) ci[r] -= d * cj[r];
    }
    const n = Math.sqrt(form(k, ci, ci));
    for (let r = 0; r < 4; r++) ci[r] /= n;
  }
  return m;
}

// ---------------------------------------------------------------------------
// Ray state and transport
// ---------------------------------------------------------------------------
export interface Ray { o: V4; v: V4 }

/** Advance a ray by arc length t: new origin γ(t), new direction γ'(t) (Eqs. 3–4). */
export function advance(k: Kappa, ray: Ray, t: number): Ray {
  return { o: geodesic(k, ray.o, ray.v, t), v: geodesicDir(k, ray.o, ray.v, t) };
}

/** Apply an isometry to a ray (points and tangents transform the same way). */
export function transformRay(m: M4, ray: Ray): Ray {
  return { o: apply(m, ray.o), v: apply(m, ray.v) };
}

/** Make v a unit tangent at x: v ← v − κ⟨v,x⟩x, normalised (E³: v0 = 0, normalise v̄). */
export function tangentialize(k: Kappa, x: V4, v: V4): V4 {
  if (k === 0) {
    v[0] = 0;
    const n = Math.hypot(v[1], v[2], v[3]);
    for (let i = 1; i < 4; i++) v[i] /= n;
    return v;
  }
  const d = k * form(k, v, x);
  for (let i = 0; i < 4; i++) v[i] -= d * x[i];
  const n = Math.sqrt(form(k, v, v));
  for (let i = 0; i < 4; i++) v[i] /= n;
  return v;
}

/** Projective (Klein / gnomonic) coordinates y = x̄ / x0 (§3.5); map inset uses §4's y = x̄/(1+x0). */
export function toProjective(x: V4): Float64Array {
  return new Float64Array([x[1] / x[0], x[2] / x[0], x[3] / x[0]]);
}
export function toConformal(x: V4): Float64Array {
  const w = 1 + x[0];
  return new Float64Array([x[1] / w, x[2] / w, x[3] / w]);
}

/** Area of a geodesic sphere of radius R: 4π sn_κ(R)² (§3.9 flux test). */
export function sphereArea(k: Kappa, R: number): number {
  const s = sn(k, R);
  return 4 * Math.PI * s * s;
}
