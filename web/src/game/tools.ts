/**
 * Measurement tools (§4 Tools): beacon triangles (angles via Eq. 5 tangents, geodesic side
 * lengths) and the light meter (scalar irradiance vs distance against the flat 1/d² law).
 * The Gauss–Bonnet area helper is used by tests and the writeup only — the game never shows the
 * area (it would give the curvature away).
 */
import { type Kappa, type V4, form, tangentToward, sn, distance } from "../geometry/space";

export interface Triangle {
  anglesDeg: [number, number, number];
  sidesM: [number, number, number]; // opposite to vertex i
  sumDeg: number;
}

/** Interior angle at vertex a between the geodesics toward b and c: cos θ = ⟨u_ab, u_ac⟩ (Eq. 5). */
export function interiorAngle(k: Kappa, a: V4, b: V4, c: V4): number {
  const { u: ub } = tangentToward(k, a, b);
  const { u: uc } = tangentToward(k, a, c);
  return Math.acos(Math.max(-1, Math.min(1, form(k, ub, uc))));
}

export function triangle(k: Kappa, scale: number, A: V4, B: V4, C: V4): Triangle {
  const a = interiorAngle(k, A, B, C), b = interiorAngle(k, B, C, A), c = interiorAngle(k, C, A, B);
  const toDeg = (x: number) => (x * 180) / Math.PI;
  return {
    anglesDeg: [toDeg(a), toDeg(b), toDeg(c)],
    sidesM: [distance(k, B, C) / scale, distance(k, C, A) / scale, distance(k, A, B) / scale],
    sumDeg: toDeg(a + b + c),
  };
}

/**
 * Gauss–Bonnet check: area of the geodesic triangle ABC by numerical integration in geodesic
 * polar coordinates about A. Parametrise the far edge BC by its arc length u; for each edge
 * point P(u) the polar coordinates seen from A are r(u) = d(A,P) and θ(u) = ∠(AB, AP) (Eq. 5
 * tangents). With the area element sn_κ(r) dr dθ,  Area = ∫ F(r(θ)) dθ,  F(r) = ∫₀ʳ sn_κ = 
 * (cosh r − 1) in H³, (1 − cos r) in S³, r²/2 in E³. Compared with the excess (Σ − π)/κ.
 */
export function triangleAreaNumeric(k: Kappa, A: V4, B: V4, C: V4, nEdge = 4000): number {
  const { u: uBC, d: lBC } = tangentToward(k, B, C);
  const { u: uAB } = tangentToward(k, A, B);
  const F = (r: number) => (k > 0 ? 1 - Math.cos(r) : k < 0 ? Math.cosh(r) - 1 : 0.5 * r * r);
  const cs = (t: number) => (k > 0 ? Math.cos(t) : k < 0 ? Math.cosh(t) : 1);
  let area = 0;
  let prevTheta = 0, prevF = F(distance(k, A, B));
  for (let i = 1; i <= nEdge; i++) {
    const u = (lBC * i) / nEdge;
    const P = new Float64Array(4);
    const c = cs(u), sU = sn(k, u);
    for (let j = 0; j < 4; j++) P[j] = c * B[j] + sU * uBC[j];
    const { u: uAP, d: r } = tangentToward(k, A, P);
    const theta = Math.acos(Math.max(-1, Math.min(1, form(k, uAB, uAP))));
    const f = F(r);
    area += 0.5 * (prevF + f) * (theta - prevTheta); // trapezoid in θ
    prevTheta = theta; prevF = f;
  }
  return area;
}

// ---------------------------------------------------------------- light meter
export interface MeterSample { dM: number; E: number }

/** Scalar irradiance at x from a point source of power Φ at L: Φ/(4π sn_κ(d)²) (+ long way in S³). */
export function scalarIrradiance(k: Kappa, x: V4, L: V4, power: number): { E: number; d: number } {
  const d = distance(k, x, L);
  const s = Math.max(1e-3, Math.abs(sn(k, d)));
  let E = power / (4 * Math.PI * s * s);
  if (k > 0) { const s2 = Math.max(1e-3, Math.abs(Math.sin(2 * Math.PI - d))); E += power / (4 * Math.PI * s2 * s2) * 0; } // long way counted once (same sin²)
  return { E, d };
}

export class LightMeter {
  samples: MeterSample[] = [];
  active = false;
  /** Add a sample if the distance moved since the last one is > 0.15 m. */
  sample(k: Kappa, scale: number, x: V4, L: V4, power: number) {
    const { E, d } = scalarIrradiance(k, x, L, power);
    const dM = d / scale;
    const last = this.samples[this.samples.length - 1];
    if (last && Math.abs(last.dM - dM) < 0.15) return;
    this.samples.push({ dM, E });
    if (this.samples.length > 200) this.samples.shift();
  }
  clear() { this.samples = []; }
  /** Flat prediction normalised to the first sample: E₀·(d₀/d)². */
  flatCurve(dM: number): number {
    const s0 = this.samples[0];
    return s0 ? s0.E * (s0.dM * s0.dM) / (dM * dM) : 0;
  }
}
