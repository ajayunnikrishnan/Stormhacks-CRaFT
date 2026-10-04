/**
 * Camera-at-the-origin navigation (§3.7). The world is carried by the isometry W
 * (world → camera). We keep a "body" isometry Wb (position + yaw) and a separate pitch:
 *   W = R_pitch · Wb
 * Movement:  Wb ← T(−δ·dir) · Wb   (dir in body coordinates, horizontal)
 * Yaw:       Wb ← R_y(Δ) · Wb
 * The body stays on the totally geodesic "eye plane" x2 = 0 because every move is a
 * translation along a tangent of that plane or a rotation about its normal (§4).
 * Wb is re-orthonormalised (J-Gram–Schmidt) every frame.
 */
import {
  type Kappa, type M4, type V4, ORIGIN, apply, identity, inverse, mul, reorthonormalize, rotation,
  translationByVector, translationTo, embedPoint, m4, logAtOrigin,
} from "../geometry/space";
import type { RepoCamera } from "../foam/scene";

function rotY(a: number): number[] { const c = Math.cos(a), s = Math.sin(a); return [c, 0, -s, 0, 1, 0, s, 0, c]; } // column-major
function rotX(a: number): number[] { const c = Math.cos(a), s = Math.sin(a); return [1, 0, 0, 0, c, s, 0, -s, c]; }

export class IsoCamera {
  kappa: Kappa = 0;
  Wb: M4 = identity();
  pitch = 0;
  fovDeg = 70;
  /** metres per second in physical units; converted with the scene scale */
  speedMps = 2.0;
  private keys = new Set<string>();
  private dragging = false;
  private lastX = 0;
  private lastY = 0;

  attach(el: HTMLElement) {
    window.addEventListener("keydown", (e) => { if (!(e.target instanceof HTMLInputElement)) this.keys.add(e.code); });
    window.addEventListener("keyup", (e) => this.keys.delete(e.code));
    window.addEventListener("blur", () => this.keys.clear());
    el.addEventListener("mousedown", (e) => { this.dragging = true; this.lastX = e.clientX; this.lastY = e.clientY; });
    window.addEventListener("mouseup", () => (this.dragging = false));
    window.addEventListener("mousemove", (e) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastX, dy = e.clientY - this.lastY;
      this.lastX = e.clientX; this.lastY = e.clientY;
      this.yawBy(dx * 0.003);
      this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch - dy * 0.003));
    });
  }

  /** Reset to a world position (model coords) with given yaw; pitch 0. */
  setPose(kappa: Kappa, posModel: V4, yaw: number) {
    this.kappa = kappa;
    const T = translationTo(kappa, posModel); // o → pos
    this.Wb = mul(rotation(rotY(yaw)), inverse(kappa, T));
    this.pitch = 0;
  }

  /**
   * Pose from a repo camera (eye/right/up in metres) for a given (κ, s, centre):
   * W = R · T⁻¹ with R rows (r̂, û, −f̂), T = translation to embed(s·(eye − centre)).
   * For κ = 0 this reproduces the flat renderer's rays exactly.
   */
  setFromRepoCamera(c: RepoCamera, kappa: Kappa, scale: number, centre: ArrayLike<number>) {
    this.kappa = kappa;
    const r = norm3(c.right), u = norm3(c.up);
    const f = norm3(cross3(u, r)); // camera.py: forward = normalize(cross(up, right))
    // column-major R3 with rows r, u, −f  ⇒ R3[col*3+row]
    const R3 = [r[0], u[0], -f[0], r[1], u[1], -f[1], r[2], u[2], -f[2]];
    const pos = embedPoint(kappa, [(c.eye[0] - centre[0]) * scale, (c.eye[1] - centre[1]) * scale, (c.eye[2] - centre[2]) * scale]);
    this.Wb = mul(rotation(R3), inverse(kappa, translationTo(kappa, pos)));
    this.pitch = 0;
    const ty = Math.hypot(c.up[0], c.up[1], c.up[2]);
    this.fovDeg = (2 * Math.atan(ty) * 180) / Math.PI;
  }

  /** Turn the view right by a (radians): the camera rotates by Q = R_y(−a), so W ← Q⁻¹W = R_y(a)·W. */
  yawBy(a: number) { this.Wb = mul(rotation(rotY(a)), this.Wb); }

  /** Body yaw: angle of the body's forward (−z) in world tangent coordinates at the position. */
  yaw(): number {
    // forward in world coords = invWb · (0,0,0,−1); take its spatial part transported to the origin ≈ spatial part
    const f = apply(inverse(this.kappa, this.Wb), new Float64Array([0, 0, 0, -1]));
    return Math.atan2(-f[1], -f[3]);
  }

  /** Physical position in metres (scene frame): log map at the origin / s + centre. */
  physicalPos(kappa: Kappa, scale: number, centre: ArrayLike<number>): Float64Array {
    const p = this.worldPos();
    const l = logAtOrigin(kappa, p);
    return new Float64Array([l[0] / scale + centre[0], l[1] / scale + centre[1], l[2] / scale + centre[2]]);
  }

  /** Set pose from physical metres + yaw for a new (κ, s). Keeps the body on the eye plane. */
  setPoseMetres(kappa: Kappa, scale: number, centre: ArrayLike<number>, posM: ArrayLike<number>, yaw: number) {
    const pos = embedPoint(kappa, [(posM[0] - centre[0]) * scale, (posM[1] - centre[1]) * scale, (posM[2] - centre[2]) * scale]);
    const pitch = this.pitch;
    this.setPose(kappa, pos, yaw);
    this.pitch = pitch;
  }

  /** Translate the body by a camera-space vector (model units). */
  moveBy(dBody: ArrayLike<number>) {
    // moving the camera by +d means the world moves by −d:  Wb ← T(−d) · Wb
    const T = translationByVector(this.kappa, [-dBody[0], -dBody[1], -dBody[2]]);
    this.Wb = mul(T, this.Wb);
  }

  /** Per-frame input: WASD in the horizontal body plane, Q/E vertical (free mode). dt in seconds. */
  update(dt: number, scale: number, allowVertical = true) {
    const sp = this.speedMps * scale * (this.keys.has("ShiftLeft") || this.keys.has("ShiftRight") ? 4 : 1) * dt;
    const d = [0, 0, 0];
    if (this.keys.has("KeyW")) d[2] -= sp;
    if (this.keys.has("KeyS")) d[2] += sp;
    if (this.keys.has("KeyD")) d[0] += sp;
    if (this.keys.has("KeyA")) d[0] -= sp;
    if (allowVertical) {
      if (this.keys.has("KeyE") || this.keys.has("Space")) d[1] += sp;
      if (this.keys.has("KeyQ") || this.keys.has("ControlLeft")) d[1] -= sp;
    }
    if (d[0] || d[1] || d[2]) this.moveBy(d);
    reorthonormalize(this.kappa, this.Wb);
  }

  /** world → camera isometry including pitch */
  W(): M4 { return mul(rotation(rotX(-this.pitch)), this.Wb); } // +pitch looks up
  /** camera → world */
  invW(): M4 { return inverse(this.kappa, this.W()); }
  /** camera position in world model coordinates */
  worldPos(): V4 { return apply(inverse(this.kappa, this.Wb), ORIGIN); }

  tanHalfFov(aspect: number): [number, number] {
    const ty = Math.tan((this.fovDeg * Math.PI) / 360);
    return [ty * aspect, ty];
  }
}

function cross3(a: ArrayLike<number>, b: ArrayLike<number>) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function norm3(a: ArrayLike<number>) { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
export const _m4 = m4;
