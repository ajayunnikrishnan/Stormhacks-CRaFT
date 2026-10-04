/**
 * Per-curvature site data (§3.4–3.6): embeds the flat scene for the current (κ, s) and
 * produces the a_i / (R_i, cs R_i) arrays the curved shaders read. Recomputed on the CPU
 * whenever the curvature slider moves (~N·30 flops, well under a millisecond per 25k cells).
 */
import { type Kappa, cs, embedPoint, v4 } from "../geometry/space";
import type { FoamScene } from "./scene";

export interface CurvatureParams {
  /** curvature in 1/m²; sign gives κ, s = √|k| */
  k: number;
  kappa: Kappa;
  scale: number;
}

export const E3_EPS = 1e-9;

export function curvatureParams(k: number): CurvatureParams {
  if (Math.abs(k) < E3_EPS) return { k: 0, kappa: 0, scale: 1 };
  return { k, kappa: k > 0 ? 1 : -1, scale: Math.sqrt(Math.abs(k)) };
}

export interface CurvedSites {
  /** A holds (a0 − 1, ā) for κ≠0 and (−pm, p̄) for E³; shaders add the 1 back where p_i is needed. */
  params: CurvatureParams;
  centre: Float64Array; // scene centre (metres) mapped to the model origin
  A: Float32Array; // [N,4]
  rad: Float32Array; // [N,2] (R, cs R)
  maxR: number;
}

/** Centre used for embedding: the manifest's curved.centre (bbox centre of scene cells). */
export function sceneCentre(scene: FoamScene): Float64Array {
  const c = (scene.manifest as unknown as { curved?: { centre: number[] } }).curved?.centre;
  if (c) return new Float64Array(c);
  const lo = scene.manifest.bbox_min, hi = scene.manifest.bbox_max;
  return new Float64Array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])]);
}

export function computeCurvedSites(scene: FoamScene, params: CurvatureParams, centre: Float64Array): CurvedSites {
  const n = scene.n, k = params.kappa, s = params.scale;
  const A = new Float32Array(n * 4), rad = new Float32Array(n * 2);
  const p = scene.pos;
  const tmp = v4();
  let maxR = 0;
  for (let i = 0, o = 0; i < n; i++, o += 4) {
    const x = (p[o] - centre[0]) * s, y = (p[o + 1] - centre[1]) * s, z = (p[o + 2] - centre[2]) * s;
    const R = p[o + 3] * s;
    if (R > maxR) maxR = R;
    if (k === 0) {
      // E³: a = (−pm, p̄), pm = ½(|p̄|² − r²)  (so that argmax ⟨x,a⟩' = argmin power distance)
      A[o] = -0.5 * (x * x + y * y + z * z - R * R);
      A[o + 1] = x; A[o + 2] = y; A[o + 3] = z;
      rad[2 * i] = R; rad[2 * i + 1] = 1;
    } else {
      embedPoint(k, [x, y, z], tmp);
      const c = cs(k, R);
      // store a0 − 1 so that neighbour differences Δa0 = O(s²) keep their fp32 digits
      A[o] = (tmp[0] - c) / c; A[o + 1] = tmp[1] / c; A[o + 2] = tmp[2] / c; A[o + 3] = tmp[3] / c;
      rad[2 * i] = R; rad[2 * i + 1] = c;
    }
  }
  if (k > 0 && maxR >= Math.PI / 2) throw new Error(`S³ needs every radius < π/2 (max ${maxR.toFixed(3)}); lower k`);
  return { params, centre, A, rad, maxR };
}

/** Start cell for a camera at world model point oW: argmax_i ⟨oW, a_i⟩' (unified cell rule). */
export function startCellCurved(sites: CurvedSites, oW: ArrayLike<number>): number {
  const kk = sites.params.kappa === 0 ? 1 : sites.params.kappa;
  const A = sites.A;
  const w0 = kk * oW[0], w1 = oW[1], w2 = oW[2], w3 = oW[3];
  let best = 0, bestV = -Infinity;
  for (let i = 0, o = 0; o < A.length; i++, o += 4) {
    const v = w0 * A[o] + w1 * A[o + 1] + w2 * A[o + 2] + w3 * A[o + 3];
    if (v > bestV) { bestV = v; best = i; }
  }
  return best;
}
