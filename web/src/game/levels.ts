/**
 * Levels (§4): each level sets the space, the available tools and a goal predicate.
 *   1 flat tutorial · 2 "The lamps are dying" (H³) · 3 "The light comes home" (S³)
 *   4 final exam (hidden weak curvature, fog) · sandbox
 * Difficulty lever (§4): strong curvature + clear view early; weak curvature + fog + dark later.
 */
import type { V4 } from "../geometry/space";
import type { DomainId } from "../topology/domain";

export type ToolId = "beacons" | "meter" | "laser" | "compass" | "map" | "lamps" | "flares" | "flashlight" | "slider" | "topology";

export interface LevelEnv {
  k: number; // curvature 1/m² (hidden in the exam)
  topology: DomainId;
  ambient: number;
  fogSigma: number;
  flashlight: boolean;
  lampBudget: number; // −1 = unlimited
  tools: ToolId[];
  hideCurvature: boolean;
  spawn?: number; // camera index
}

export interface LevelState {
  beacons: V4[];
  meterSamples: number;
  lampsPlaced: number;
  laserUsed: boolean;
  compassLoopDeg: number;
  targetIrradiance: number; // W/m² at the level target (0 if none)
  submittedK: number | null;
  elapsed: number;
}

export interface Level {
  id: string;
  title: string;
  goal: string;
  lesson: string;
  env: LevelEnv;
  /** target position in metres (scene frame) for lighting puzzles */
  targetM?: [number, number, number];
  targetThreshold?: number;
  /** returns progress text and done flag */
  check(st: LevelState): { done: boolean; progress: string };
}

const ALL_TOOLS: ToolId[] = ["beacons", "meter", "laser", "compass", "map", "lamps", "flares", "flashlight", "slider", "topology"];

export const LEVELS: Level[] = [
  {
    id: "tutorial",
    title: "1 · A normal room",
    goal: "Learn the tools: place 3 beacons (B) and read the triangle; walk away from a lamp (P) with the light meter (I) on; fire the laser (L).",
    lesson: "In flat space a triangle's angles sum to 180° and light fades as 1/d². Remember what 'normal' looks like.",
    env: { k: 0, topology: "none", ambient: 0.35, fogSigma: 0.02, flashlight: true, lampBudget: -1, tools: ALL_TOOLS.filter((t) => t !== "slider" && t !== "topology"), hideCurvature: false, spawn: 0 },
    check(st) {
      const a = st.beacons.length >= 3, b = st.meterSamples >= 6, c = st.laserUsed;
      return { done: a && b && c, progress: `beacons ${a ? "✓" : `${st.beacons.length}/3`} · meter ${b ? "✓" : `${st.meterSamples}/6 samples`} · laser ${c ? "✓" : "–"}` };
    },
  },
  {
    id: "lamps",
    title: "2 · The lamps are dying",
    goal: "This space is hyperbolic. Light dies faster than 1/d². Light the marker (★) to 0.8 W/m² with at most 3 lamps (P).",
    lesson: "Hyperbolic space has exponentially more room at distance: a lamp's light spreads over area 4π sinh²(d) instead of 4πd².",
    env: { k: -0.12, topology: "none", ambient: 0.06, fogSigma: 0.05, flashlight: false, lampBudget: 3, tools: ["beacons", "meter", "compass", "map", "lamps", "laser"], hideCurvature: false, spawn: 0 },
    targetM: [2.6, 1.3, -1.0],
    targetThreshold: 0.8,
    check(st) { return { done: st.targetIrradiance >= 0.8, progress: `marker irradiance ${st.targetIrradiance.toFixed(2)} / 0.80 W/m² · lamps ${st.lampsPlaced}/3` }; },
  },
  {
    id: "home",
    title: "3 · The light comes home",
    goal: "This space is spherical. The marker (★) floats 9 m out over the void — unreachable, and too far for a lamp at the edge. Light it anyway: in S³ a lamp's light reconverges at the lamp's antipode, half-way round the universe (15.3 m here). Find the spot on the floor that is the marker's antipode and put a lamp there (P places one 1.5 m ahead of you).",
    lesson: "In S³ every geodesic from a point meets again at its antipode (distance πR). Irradiance Φ/(4π sin²(d/R)) blows up as d → πR: light 'comes home'. Walk straight away from the marker and keep going.",
    env: { k: 0.042, topology: "none", ambient: 0.05, fogSigma: 0.0, flashlight: true, lampBudget: 2, tools: ["beacons", "meter", "compass", "map", "lamps", "laser"], hideCurvature: false, spawn: 3 },
    targetM: [13.0, 1.6, -1.5],
    targetThreshold: 0.5,
    check(st) { return { done: st.targetIrradiance >= 0.5, progress: `marker irradiance ${st.targetIrradiance.toFixed(2)} / 0.50 W/m² · lamps ${st.lampsPlaced}/2` }; },
  },
  {
    id: "exam",
    title: "4 · Final exam: what shape is this space?",
    goal: "The curvature is hidden and weak, and it is foggy. Measure it (triangles, light meter, compass loops), then dial your estimate on the slider and press Enter.",
    lesson: "Angle excess = κ·Area. Irradiance = Φ/(4π sn_κ(d)²). Holonomy after a loop = κ·enclosed area. Any one of them gives κ.",
    env: { k: 0, topology: "none", ambient: 0.15, fogSigma: 0.09, flashlight: true, lampBudget: -1, tools: ["beacons", "meter", "laser", "compass", "map", "lamps", "flashlight", "slider"], hideCurvature: true, spawn: 0 },
    check(st) { return { done: st.submittedK !== null, progress: st.submittedK === null ? "measure, then submit with Enter" : "submitted" }; },
  },
  {
    id: "sandbox",
    title: "Sandbox",
    goal: "Everything unlocked: curvature slider, topology gallery, all tools. Lesson cards appear when you pick a space.",
    lesson: "",
    env: { k: 0, topology: "none", ambient: 0.25, fogSigma: 0.04, flashlight: true, lampBudget: -1, tools: ALL_TOOLS, hideCurvature: false, spawn: 0 },
    check() { return { done: false, progress: "" }; },
  },
];

/** Hidden curvature for the exam: weak, random sign, reproducible per session seed. */
export function examCurvature(seed: number): number {
  let a = seed >>> 0;
  const r = () => { a = (a + 0x6d2b79f5) >>> 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const mag = 0.006 + r() * 0.03;
  return r() < 0.5 ? -mag : mag;
}

/** Score 0–100 from the relative error of the submitted curvature (log-scale aware). */
export function examScore(trueK: number, guessK: number): { score: number; errorPct: number } {
  const sameSign = Math.sign(trueK) === Math.sign(guessK);
  const errorPct = (Math.abs(guessK - trueK) / Math.abs(trueK)) * 100;
  let score = Math.max(0, 100 - errorPct);
  if (!sameSign) score = Math.min(score, 20);
  return { score: Math.round(score), errorPct };
}

export const LESSON_CARDS: Record<DomainId, string> = {
  none: "Open space: no walls, just the geometry. Use the slider to bend it.",
  torus3: "3-torus: a box whose opposite faces are glued. Walk out of one wall and you return through the opposite wall, unchanged. Finite volume, no edges.",
  halfturn: "Half-turn space: like the 3-torus, but one pair of faces is glued with a 180° twist. Your own copy ahead is upside down.",
  klein: "Klein space: one pair of faces glued with a mirror. Walk through and you come back left-handed — a non-orientable universe.",
  e434: "{4,3,4}: the ordinary cubic tiling. Four cubes meet at each edge, eight at each corner.",
  h435: "{4,3,5}: five cubes meet at every edge — only possible because hyperbolic cubes have 72° dihedral angles. Count them along an edge.",
  s433: "{4,3,3}: eight cubes tile the whole 3-sphere (the tesseract's cells). Three cubes meet at every edge, and the far side of the universe is just a few cubes away.",
  pds: "Poincaré dodecahedral space: a dodecahedron whose opposite faces are glued with a 36° twist. Finite, spherical, and once a candidate for the shape of our own universe.",
  sw: "Seifert–Weber space: a hyperbolic dodecahedron (72° dihedral angles) glued with a 108° twist. Finite volume, negatively curved.",
};
