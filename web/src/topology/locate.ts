/**
 * Point location after a domain crossing (§3.8). For each domain face we precompute a
 * LOC_GRID×LOC_GRID chart of cell IDs (exp-map coordinates at the face centre); the shader
 * takes the grid entry as a guess and finishes with a short steepest-ascent walk on ⟨x,a_i⟩'.
 * The same ascent (`descend`) is used here on the CPU to build the grid with warm starts.
 */
import { type V4, apply, embedPoint, v4, logAtOrigin } from "../geometry/space";
import type { CurvedSites } from "../foam/curved";
import type { Domain } from "./domain";

export const LOC_GRID = 32;

/** ⟨x, a_c⟩' with the stored a (constant shifts cancel in comparisons). */
export function cellValue(A: Float32Array, kk: number, c: number, x: V4): number {
  const o = c * 4;
  return kk * A[o] * x[0] + A[o + 1] * x[1] + A[o + 2] * x[2] + A[o + 3] * x[3];
}

/** Steepest ascent over the adjacency from `start`; returns the cell containing x. */
export function descend(A: Float32Array, kk: number, adjOff: Uint32Array, adjIdx: Uint32Array, start: number, x: V4, maxIter = 100000): { cell: number; steps: number } {
  let c = start, best = cellValue(A, kk, c, x), steps = 0;
  for (; steps < maxIter; steps++) {
    const a = adjOff[c], b = adjOff[c + 1];
    let bj = -1;
    for (let q = a; q < b; q++) {
      const j = adjIdx[q];
      const v = cellValue(A, kk, j, x);
      if (v > best + 1e-9) { best = v; bj = j; }
    }
    if (bj < 0) break;
    c = bj;
  }
  return { cell: c, steps };
}

export function bruteForceCell(A: Float32Array, kk: number, x: V4): number {
  let best = -1, bv = -Infinity;
  for (let c = 0, o = 0; o < A.length; c++, o += 4) {
    const v = kk * A[o] * x[0] + A[o + 1] * x[1] + A[o + 2] * x[2] + A[o + 3] * x[3];
    if (v > bv) { bv = v; best = c; }
  }
  return best;
}

export interface LocateGrid { ids: Uint32Array; chartHalf: Float32Array; grid: number }

/** Chart half-extent per face: max in-plane log distance of the face's vertices (×1.05). */
export function chartHalfExtents(d: Domain): Float32Array {
  const out = new Float32Array(d.faces.length);
  const k = d.kappa;
  for (let f = 0; f < d.faces.length; f++) {
    const F = d.faces[f];
    let m = 0;
    for (const x of d.vertices) {
      const w = F.w;
      if (Math.abs(w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3]) > 1e-7) continue; // not on this face
      const l = logAtOrigin(k, apply(F.invFrame, x));
      m = Math.max(m, Math.hypot(l[0], l[1]));
    }
    if (m === 0) throw new Error(`face ${f} has no vertices`);
    out[f] = m * 1.05;
  }
  return out;
}

/** Grid point (i,j) of face f in world model coords. */
export function gridPoint(d: Domain, chartHalf: Float32Array, f: number, i: number, j: number, grid = LOC_GRID): V4 {
  const u = ((i + 0.5) / grid * 2 - 1) * chartHalf[f], w = ((j + 0.5) / grid * 2 - 1) * chartHalf[f];
  return apply(d.faces[f].frame, embedPoint(d.kappa, [u, w, 0]));
}

export function buildLocateGrid(d: Domain, sites: CurvedSites, adjOff: Uint32Array, adjIdx: Uint32Array, previous?: LocateGrid): LocateGrid {
  const grid = LOC_GRID;
  const kk = sites.params.kappa === 0 ? 1 : sites.params.kappa;
  const chartHalf = chartHalfExtents(d);
  const ids = new Uint32Array(d.faces.length * grid * grid);
  for (let f = 0; f < d.faces.length; f++) {
    let warm = previous ? previous.ids[f * grid * grid] : bruteForceCell(sites.A, kk, gridPoint(d, chartHalf, f, 0, 0));
    for (let j = 0; j < grid; j++) {
      for (let i = 0; i < grid; i++) {
        const idx = f * grid * grid + j * grid + i;
        const start = previous ? previous.ids[idx] : warm;
        const x = gridPoint(d, chartHalf, f, i, j);
        const r = descend(sites.A, kk, adjOff, adjIdx, start, x);
        ids[idx] = r.cell;
        warm = r.cell;
      }
      if (!previous) warm = ids[f * grid * grid + j * grid]; // start next row from this row's first cell
    }
  }
  return { ids, chartHalf, grid };
}
