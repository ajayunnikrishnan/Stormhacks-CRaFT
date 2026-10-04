import { describe, expect, it } from "vitest";
import { type Kappa, embedPoint, ORIGIN } from "../src/geometry/space";
import { triangle, triangleAreaNumeric, scalarIrradiance } from "../src/game/tools";

describe("beacon triangle (§4 Tools) and Gauss–Bonnet (§7 validation)", () => {
  it("flat: angles sum to 180°, sides in metres", () => {
    const A = embedPoint(0, [0, 0, 0]), B = embedPoint(0, [3, 0, 0]), C = embedPoint(0, [0, 0, 4]);
    const t = triangle(0, 1, A, B, C);
    expect(t.sumDeg).toBeCloseTo(180, 9);
    expect(t.anglesDeg[0]).toBeCloseTo(90, 9);
    expect(t.sidesM[0]).toBeCloseTo(5, 9);
  });
  for (const k of [-1, 1] as Kappa[])
    it(`κ=${k}: numerically integrated area × κ == angle excess (Σ − π)`, () => {
      for (const sz of [0.3, 0.6, 0.9]) {
        const A = ORIGIN, B = embedPoint(k, [sz, 0, 0.1 * sz]), C = embedPoint(k, [0.2 * sz, 0, sz]);
        const t = triangle(k, 1, A, B, C);
        const excess = (t.sumDeg * Math.PI) / 180 - Math.PI; // = κ·Area
        const area = triangleAreaNumeric(k, A, B, C);
        expect(Math.abs(k * area - excess)).toBeLessThan(2e-3 * Math.max(1, Math.abs(excess) * 10));
        if (k < 0) expect(t.sumDeg).toBeLessThan(180); else expect(t.sumDeg).toBeGreaterThan(180);
      }
    });
  it("light meter: in H³ irradiance falls below 1/d², in S³ above, flat exactly 1/d²", () => {
    for (const k of [-1, 0, 1] as Kappa[]) {
      const L = ORIGIN;
      const E1 = scalarIrradiance(k, embedPoint(k, [0.5, 0, 0]), L, 1).E;
      const E2 = scalarIrradiance(k, embedPoint(k, [1.0, 0, 0]), L, 1).E;
      const ratio = E2 / E1; // flat: 0.25
      if (k === 0) expect(ratio).toBeCloseTo(0.25, 9);
      if (k < 0) expect(ratio).toBeLessThan(0.25);
      if (k > 0) expect(ratio).toBeGreaterThan(0.25);
    }
  });
});
