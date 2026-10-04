/**
 * Player state on top of IsoCamera (§4 Navigation):
 *  - cell tracking by steepest ascent (no per-frame brute force),
 *  - collision: a short geodesic probe against the foam's dense regions,
 *  - compass: a tangent vector parallel-transported by every movement isometry (§4 Tools).
 *
 * Compass maths. In camera coordinates (camera at the origin) a translation moves the frame
 * and the vector identically, so parallel transport along the player's own geodesic leaves
 * the compass components unchanged; a yaw by R rotates them by R. After a closed loop the
 * compass therefore differs from the heading by the holonomy angle −κ·A (Gauss–Bonnet),
 * which is exactly what the tool is meant to show (tests/compass.test.ts).
 */
import { type Kappa, type V4, type M4, apply, form, inverse, rotation, mul, geodesic, tangentToward, distance, ORIGIN, v4, translationByVector, reorthonormalize, logAtOrigin, embedPoint } from "../geometry/space";
import type { SceneArrays } from "../foam/sceneData";
import { halfToFloat } from "../foam/sceneData";
import type { CurvedSites } from "../foam/curved";
import { descend, bruteForceCell } from "../topology/locate";
import type { IsoCamera } from "./isocamera";
import type { Domain } from "../topology/domain";

export interface CellInfo { normal: Float32Array; sigma: Float32Array }

/** Decode the fp16 (normal, σ) texture once for CPU use. */
export function decodeCellInfo(scene: SceneArrays): CellInfo {
  const n = scene.n;
  const normal = new Float32Array(n * 3), sigma = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    normal[3 * i] = halfToFloat(scene.nsigma[4 * i]);
    normal[3 * i + 1] = halfToFloat(scene.nsigma[4 * i + 1]);
    normal[3 * i + 2] = halfToFloat(scene.nsigma[4 * i + 2]);
    sigma[i] = halfToFloat(scene.nsigma[4 * i + 3]);
  }
  return { normal, sigma };
}

export class Player {
  cell = -1;
  /** compass in body coordinates (unit 3-vector; starts pointing forward = −z) */
  compass = new Float64Array([0, 0, -1]);
  radiusM = 0.25; // collision radius, metres
  collisions = true;
  /** require dense foam under the feet: no walking off the platform */
  needFloor = true;
  /** walkable region (metres, scene frame): inside it ground is known to exist, so a missed floor
   *  probe (gaps between the balls of a sparse real-capture ground) never blocks a step */
  walkBox: [number[], number[]] | null = null;
  /** floor height (metres, scene frame) and the embedding centre, set by the app */
  floorY = 0;
  centre: ArrayLike<number> = [0, 0, 0];
  private info: CellInfo;

  constructor(readonly scene: SceneArrays, info?: CellInfo) {
    this.info = info ?? decodeCellInfo(scene);
  }

  private adj(): [Uint32Array, Uint32Array] { return [this.scene.adjOffU ?? this.scene.adjOff, this.scene.adjIdxU ?? this.scene.adjIdx]; }

  /** Cell containing x (model coords), warm-started from the last known cell. */
  locate(sites: CurvedSites, x: V4): number {
    const kk = sites.params.kappa === 0 ? 1 : sites.params.kappa;
    const [off, idx] = this.adj();
    if (this.cell < 0) this.cell = bruteForceCell(sites.A, kk, x);
    this.cell = descend(sites.A, kk, off, idx, this.cell, x).cell;
    return this.cell;
  }

  /**
   * Is model point x inside dense foam? Inside the cell's ball AND behind the (undisplaced)
   * dipole plane with σ above threshold. Steiner / empty cells never block.
   */
  /** why the last tryMove refused (debug hook) */
  lastBlock = "";

  /**
   * @param shell  false (collision): the whole ball of a dense cell counts, so a thin plank is as
   *               thick as its cells (a step cannot slip through between two probes);
   *               true (floor probe): only the dense half behind the dipole plane
   */
  isSolid(sites: CurvedSites, x: V4, cellHint?: number, shell = false): boolean {
    const k = sites.params.kappa, kk = k === 0 ? 1 : k;
    const [off, idx] = this.adj();
    const c = descend(sites.A, kk, off, idx, cellHint ?? (this.cell < 0 ? 0 : this.cell), x).cell;
    const R = sites.rad[2 * c], cR = sites.rad[2 * c + 1];
    // collision: the cell must be opaque across its ball (optical depth σ·2R); big faint cells
    // (fog, sky, floaters in real captures) are not walls. Floor probe: any dense cell.
    const tau = this.info.sigma[c] * 2 * R;
    if (shell ? this.info.sigma[c] < 1e-3 : tau < 1.0) return false;
    const a = sites.A.subarray(4 * c, 4 * c + 4);
    const P = k === 0 ? v4(1, a[1], a[2], a[3]) : v4((a[0] + 1) * cR, a[1] * cR, a[2] * cR, a[3] * cR);
    if (distance(k, x, P) > R) return false;
    if (!shell) return true;
    // dense side: ⟨x, N⟩_κ' ≤ 0 with N the transported normal; equivalently in the cell frame n̄·log(x) ≤ 0
    const n = this.info.normal.subarray(3 * c, 3 * c + 3);
    // transported normal via the translation frame: N = T_P (0, n̄)
    const T = translationToFrame(k, P);
    const N = apply(T, v4(0, n[0], n[1], n[2]));
    const val = k === 0 ? N[1] * (x[1] - P[1]) + N[2] * (x[2] - P[2]) + N[3] * (x[3] - P[3]) : form(k, x, N);
    return val <= 0;
  }

  /**
   * Try to move the body by dBody (camera coords, model units). Probes the destination and
   * a ring of points at the collision radius; slides along the blocked axis by trying x and z
   * separately. Returns true if any movement happened.
   */
  tryMove(cam: IsoCamera, sites: CurvedSites, dBody: number[], scale: number): boolean {
    if (!this.collisions) { cam.moveBy(dBody); return true; }
    const invWb = inverse(cam.kappa, cam.Wb);
    const r = this.radiusM * scale;
    // body probes: a ring at eye height plus points at chest, waist and knee height (real captures
    // have low obstacles such as benches; the knee probe stays above ground clutter)
    // a column of probes every 7.5 cm from 25 cm above the floor to the eye, swept along the step.
    // No lateral ring: cells are opaque across their whole ball for collisions, so the surface
    // itself is the wall, and a ring at the collision radius snagged on floaters in real captures.
    const ring: number[][] = [];
    // from 40 cm up: real ground is littered with ankle-height clutter cells (gravel, leaves)
    for (let h = 0.4; h <= 1.6; h += 0.075) ring.push([0, -(1.6 - h) * scale, 0]);
    void r;
    // swept: a step can be longer than a shell is thick, so sample each probe along the step
    const ringBlocked = (d: number[]) => {
      const len = Math.hypot(d[0], d[1], d[2]);
      const nSub = Math.max(1, Math.ceil(len / (0.03 * scale)));
      for (const o of ring) for (let i = 1; i <= nSub; i++) {
        const t = i / nSub;
        const p = apply(invWb, bodyPoint(cam.kappa, [d[0] * t + o[0], d[1] * t + o[1], d[2] * t + o[2]]));
        if (this.isSolid(sites, p)) { this.lastBlock = `body probe h=${(1.6 + o[1] / scale).toFixed(2)} t=${t.toFixed(2)} cell=${this.cell}`; return true; }
      }
      return false;
    };
    // already overlapping a thin shell (a step can cross it between probes)? then only the
    // destination itself must be free, so the walker can always back out instead of freezing
    const penetrating = ring.some((o) => this.isSolid(sites, apply(invWb, bodyPoint(cam.kappa, o))));
    const attempt = (d: number[]) => {
      if (!d[0] && !d[1] && !d[2]) return false;
      const dest = apply(invWb, bodyPoint(cam.kappa, d));
      if (this.isSolid(sites, dest)) { this.lastBlock = "destination solid"; return false; }
      if (!penetrating && ringBlocked(d)) return false;
      if (this.needFloor) {
        // The floor was embedded as the exp-map chart plane y = floorY (metres), so look for it
        // there: take the destination's chart coordinates and probe just below floorY. Exact in
        // every geometry (a fixed geodesic depth below the eye plane is not: in H³ the floor's
        // geodesic depth grows with distance from the centre).
        const l = logAtOrigin(cam.kappa, dest);
        const px = l[0] / scale + this.centre[0], pz = l[2] / scale + this.centre[2];
        // Scan a short vertical band around floorY: the dipole planes are geodesic planes tangent
        // to the chart plane only at their sites, so away from the centre the 9 cm slab is nudged
        // up or down by O(κ·|x|·cell size); ±10 cm covers it in every geometry.
        // Real captures: the ground is a layer of 5–10 cm cells whose balls leave gaps and whose
        // planes tilt, so probe a small cluster of columns over a wider band (±25 cm).
        let support = false;
        const dxs = [0, 0.12, -0.12], dzs = [0, 0.12, -0.12];
        for (let ix = 0; ix < 3 && !support; ix++) for (let iz = 0; iz < 3 && !support; iz++) {
          if (ix && iz) continue; // 5 columns: centre + 4 neighbours
          for (let dz = -0.25; dz <= 0.25 && !support; dz += 0.03) {
            const q = embedPoint(cam.kappa, [(px + dxs[ix] - this.centre[0]) * scale, (this.floorY + dz - this.centre[1]) * scale, (pz + dzs[iz] - this.centre[2]) * scale]);
            if (this.isSolid(sites, q, undefined, true)) support = true;
          }
        }
        const wb = this.walkBox;
        const inWalk = !!wb && px >= wb[0][0] && px <= wb[1][0] && pz >= wb[0][2] && pz <= wb[1][2];
        if (!support && !inWalk) { this.lastBlock = "no floor"; return false; }
      }
      cam.moveBy(d);
      return true;
    };
    this.lastBlock = "";
    if (attempt(dBody)) return true;
    const a = attempt([dBody[0], dBody[1], 0]);
    const b = attempt([0, dBody[1], dBody[2]]);
    return a || b;
  }

  /** Called with the yaw rotation applied to the body frame (same R as Wb ← R·Wb). */
  onYaw(R: M4) {
    const c = apply(R, v4(0, this.compass[0], this.compass[1], this.compass[2]));
    this.compass = new Float64Array([c[1], c[2], c[3]]);
  }

  /** Heading-relative compass angle (radians, + = compass points to the player's right). */
  compassAngle(): number { return Math.atan2(this.compass[0], -this.compass[2]); }

  /** Keep the player inside the domain and keep the compass consistent across pairings
   * (a pairing is an isometry of the world; in camera coordinates nothing changes). */
  recentre(cam: IsoCamera, domain: Domain | null): number {
    const f = cam.recentre(domain);
    if (f >= 0) this.cell = -1; // relocate from scratch after a hop
    return f;
  }
}

/** Model point at body-coordinate offset d from the camera (exp map at the origin). */
function bodyPoint(k: Kappa, d: number[]): V4 {
  return apply(translationByVector(k, d), ORIGIN);
}
function translationToFrame(k: Kappa, P: V4): M4 {
  // translationTo without re-importing: T_P maps o → P
  const c = k === 0 ? 1 : P[0];
  const s = Math.hypot(P[1], P[2], P[3]);
  const u = s < 1e-12 ? [0, 0, 0] : [P[1] / s, P[2] / s, P[3] / s];
  const m = new Float64Array(16);
  m[0] = c; m[1] = s * u[0]; m[2] = s * u[1]; m[3] = s * u[2];
  for (let i = 0; i < 3; i++) {
    m[4 * (i + 1)] = -k * s * u[i];
    for (let j = 0; j < 3; j++) m[4 * (i + 1) + 1 + j] = (i === j ? 1 : 0) + (c - 1) * u[i] * u[j];
  }
  return m;
}
export const _unused = { rotation, mul, geodesic, tangentToward, reorthonormalize };
