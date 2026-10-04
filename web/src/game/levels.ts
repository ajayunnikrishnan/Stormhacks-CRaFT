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
  movedM: number; // distance walked, metres
}

export interface Step { text: string; done: boolean }

export interface Level {
  id: string;
  title: string;
  goal: string;
  lesson: string;
  env: LevelEnv;
  /** target position in metres (scene frame) for lighting puzzles */
  targetM?: [number, number, number];
  targetThreshold?: number;
  /** returns the checklist and the overall done flag */
  check(st: LevelState): { done: boolean; steps: Step[] };
}

const ALL_TOOLS: ToolId[] = ["beacons", "meter", "laser", "compass", "map", "lamps", "flares", "flashlight", "slider", "topology"];

export const LEVELS: Level[] = [
  {
    id: "tutorial",
    title: "1 · A normal room",
    goal: "This room is ordinary flat space. Learn the four tools here, so you can tell when space stops being ordinary.",
    lesson: "In flat space a triangle's angles sum to 180° and light fades as 1/d². Remember what 'normal' looks like.",
    env: { k: 0, topology: "none", ambient: 0.35, fogSigma: 0.02, flashlight: true, lampBudget: -1, tools: ALL_TOOLS.filter((t) => t !== "slider" && t !== "topology"), hideCurvature: false, spawn: 0 },
    check(st) {
      const steps: Step[] = [
        { text: "Walk around: W A S D to move, drag the mouse to look", done: st.movedM > 3 },
        { text: `Press B at three different spots to drop beacons (${Math.min(3, st.beacons.length)}/3). They form a triangle; its angles appear on the left`, done: st.beacons.length >= 3 },
        { text: `Press P to place a lamp, then I to switch on the light meter, and walk away from the lamp (${Math.min(6, st.meterSamples)}/6 readings)`, done: st.meterSamples >= 6 },
        { text: "Press L to fire the laser: a straight line in this space", done: st.laserUsed },
      ];
      return { done: steps.every((x) => x.done), steps };
    },
  },
  {
    id: "lamps",
    title: "2 · The lamps are dying",
    goal: "The space has changed. Light dies faster here than in the room you came from. Get the pink ★ marker bright enough to read by.",
    lesson: "Hyperbolic space has exponentially more room at distance: a lamp's light spreads over area 4π sinh²(d) instead of 4πd².",
    env: { k: -0.12, topology: "none", ambient: 0.06, fogSigma: 0.05, flashlight: false, lampBudget: 3, tools: ["beacons", "meter", "compass", "map", "lamps", "laser"], hideCurvature: false, spawn: 0 },
    targetM: [2.6, 1.3, -1.0],
    targetThreshold: 0.8,
    check(st) {
      const steps: Step[] = [
        { text: "Try the light meter (I) next to a lamp (P) and watch how the dots fall below the grey 1/d² line", done: st.meterSamples >= 4 },
        { text: `Light the ★ marker to 0.80 W/m² — now ${st.targetIrradiance.toFixed(2)} (lamps used ${st.lampsPlaced}/3; X removes them)`, done: st.targetIrradiance >= 0.8 },
      ];
      return { done: st.targetIrradiance >= 0.8, steps };
    },
  },
  {
    id: "home",
    title: "3 · The light comes home",
    goal: "The ★ marker floats out over the void, far past the edge of the floor. No lamp at the edge can reach it. Yet there is a spot on this floor from which a lamp lights it perfectly.",
    lesson: "In S³ every geodesic from a point meets again at its antipode (distance πR). Irradiance Φ/(4π sin²(d/R)) blows up as d → πR: light 'comes home'. Walk straight away from the marker and keep going.",
    env: { k: 0.042, topology: "none", ambient: 0.05, fogSigma: 0.0, flashlight: true, lampBudget: 2, tools: ["beacons", "meter", "compass", "map", "lamps", "laser"], hideCurvature: false, spawn: 3 },
    targetM: [13.0, 1.6, -1.5],
    targetThreshold: 0.5,
    check(st) {
      const steps: Step[] = [
        { text: "Turn your back on the ★ marker and walk straight away from it. Keep going.", done: st.movedM > 8 },
        { text: `Where the whole sky seems to point at the marker, press P. Marker brightness: ${st.targetIrradiance.toFixed(2)} / 0.50 (lamps ${st.lampsPlaced}/2; X removes)`, done: st.targetIrradiance >= 0.5 },
      ];
      return { done: st.targetIrradiance >= 0.5, steps };
    },
  },
  {
    id: "exam",
    title: "4 · Final exam: what shape is this space?",
    goal: "Fog, darkness, and a curvature too weak to see. Measure it, then dial your estimate in the panel on the right and press Submit.",
    lesson: "Angle excess = κ·Area. Irradiance = Φ/(4π sn_κ(d)²). Holonomy after a loop = κ·enclosed area. Any one of them gives κ.",
    env: { k: 0, topology: "none", ambient: 0.15, fogSigma: 0.09, flashlight: true, lampBudget: -1, tools: ["beacons", "meter", "laser", "compass", "map", "lamps", "flashlight", "slider"], hideCurvature: true, spawn: 0 },
    check(st) {
      const steps: Step[] = [
        { text: "Measure: beacons (B) for a triangle's angle sum, a lamp (P) + meter (I) for the falloff, a walked loop for the compass", done: st.beacons.length >= 3 || st.meterSamples >= 6 },
        { text: "Dial your estimate in the panel on the right and press Submit", done: st.submittedK !== null },
      ];
      return { done: st.submittedK !== null, steps };
    },
  },
  {
    id: "sandbox",
    title: "Sandbox",
    goal: "Bend space with the slider on the right, or pick a closed universe. Every tool is unlocked: B beacons · P lamp · I light meter · L laser · T flare · F flashlight · M map · C compass.",
    lesson: "",
    env: { k: 0, topology: "none", ambient: 0.25, fogSigma: 0.04, flashlight: true, lampBudget: -1, tools: ALL_TOOLS, hideCurvature: false, spawn: 0 },
    check() { return { done: false, steps: [] }; },
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
