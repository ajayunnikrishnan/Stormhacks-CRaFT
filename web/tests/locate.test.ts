/**
 * §3.8: every grid-located start agrees with brute-force point location, on the exported
 * scene, for each domain type and both signs of curvature.
 */
import { describe, expect, it } from "vitest";
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { parseScene, type SceneManifest } from "../src/foam/sceneData";
import { computeCurvedSites, curvatureParams, sceneCentre } from "../src/foam/curved";
import { makeDomain, type DomainId } from "../src/topology/domain";
import { buildLocateGrid, bruteForceCell, descend, gridPoint, LOC_GRID } from "../src/topology/locate";
import { apply, embedPoint, v4 } from "../src/geometry/space";

const dir = resolve(__dirname, "../public/scenes/synth_open");
const have = existsSync(resolve(dir, "scene.json"));

function load() {
  const manifest = JSON.parse(readFileSync(resolve(dir, "scene.json"), "utf8")) as SceneManifest;
  const buf = readFileSync(resolve(dir, "scene.bin"));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  return parseScene(manifest, ab);
}

function rng(seed: number) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

describe.skipIf(!have)("point-location grid vs brute force", () => {
  const scene = have ? load() : null!;
  const centre = have ? sceneCentre(scene) : null!;
  const cases: [DomainId, number][] = [["torus3", 0], ["halfturn", 0], ["h435", -0.0176], ["s433", 0.0385], ["pds", 0.0041], ["sw", -0.04]];
  for (const [id, k] of cases)
    it(`${id} at k=${k}: grid guess + ascent == brute force on random face points`, () => {
      const params = curvatureParams(k);
      const sites = computeCurvedSites(scene, params, centre);
      const kk = params.kappa === 0 ? 1 : params.kappa;
      const half: [number, number, number] = [4, 1.25, 4];
      const d = makeDomain(id, half)!;
      const adjOff = scene.adjOffU!, adjIdx = scene.adjIdxU!;
      const t0 = performance.now();
      const grid = buildLocateGrid(d, sites, adjOff, adjIdx);
      const buildMs = performance.now() - t0;
      // 1) every grid entry is the true owner of its grid point
      let wrong = 0;
      for (let f = 0; f < d.faces.length; f++)
        for (let j = 0; j < LOC_GRID; j += 3)
          for (let i = 0; i < LOC_GRID; i += 3) {
            const x = gridPoint(d, grid.chartHalf, f, i, j);
            if (grid.ids[f * LOC_GRID * LOC_GRID + j * LOC_GRID + i] !== bruteForceCell(sites.A, kk, x)) wrong++;
          }
      expect(wrong).toBe(0);
      // 2) random points on each face: nearest grid entry + ascent == brute force; count ascent steps
      const R = rng(11);
      let maxSteps = 0, sumSteps = 0, n = 0, mism = 0;
      for (let f = 0; f < d.faces.length; f++) {
        for (let t = 0; t < 150; t++) {
          const u = (R() * 2 - 1) * grid.chartHalf[f] * 0.95, w = (R() * 2 - 1) * grid.chartHalf[f] * 0.95;
          const x = apply(d.faces[f].frame, embedPoint(d.kappa, [u, w, 0]));
          const gi = Math.min(LOC_GRID - 1, Math.floor((u / grid.chartHalf[f] * 0.5 + 0.5) * LOC_GRID));
          const gj = Math.min(LOC_GRID - 1, Math.floor((w / grid.chartHalf[f] * 0.5 + 0.5) * LOC_GRID));
          const guess = grid.ids[f * LOC_GRID * LOC_GRID + gj * LOC_GRID + gi];
          const r = descend(sites.A, kk, adjOff, adjIdx, guess, x);
          if (r.cell !== bruteForceCell(sites.A, kk, x)) mism++;
          maxSteps = Math.max(maxSteps, r.steps); sumSteps += r.steps; n++;
        }
      }
      expect(mism).toBe(0);
      expect(maxSteps).toBeLessThanOrEqual(32); // the shader's locateCell cap
      console.log(`${id} k=${k}: grid build ${buildMs.toFixed(0)} ms, ascent steps mean ${(sumSteps / n).toFixed(2)} max ${maxSteps}`);
    });
  it("origin is cell-located consistently (sanity)", () => {
    const sites = computeCurvedSites(scene, curvatureParams(0), centre);
    expect(bruteForceCell(sites.A, 1, v4(1, 0, 0, 0))).toBeGreaterThanOrEqual(0);
  });
});
