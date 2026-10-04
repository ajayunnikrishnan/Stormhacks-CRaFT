/**
 * Phase 1 fly camera in the repo's TorchCamera convention:
 *   right = tan(fovx/2) * r̂,  up = tan(fovy/2) * û,  forward = normalize(cross(up, right)).
 * (Phase 5 replaces this with the isometry-based camera at the origin, §3.7.)
 */
import type { CameraState, } from "../render/renderer";
import type { RepoCamera } from "../foam/scene";

const WORLD_UP = [0, 1, 0];

function cross(a: ArrayLike<number>, b: ArrayLike<number>): Float32Array {
  return new Float32Array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]);
}
function normalize(v: Float32Array): Float32Array {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return new Float32Array([v[0] / l, v[1] / l, v[2] / l]);
}

export class FlyCamera {
  eye = new Float32Array([0, 1.6, 0]);
  yaw = 0; // radians, 0 = looking down -z
  pitch = 0;
  fovDeg = 70;
  speed = 2.0; // m/s
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
      this.yaw -= dx * 0.003;
      this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch - dy * 0.003));
    });
  }

  /** Set pose from a repo camera (eye/right/up). */
  setFromRepoCamera(c: RepoCamera) {
    this.eye = new Float32Array(c.eye);
    const f = normalize(cross(c.up, c.right));
    this.yaw = Math.atan2(-f[0], -f[2]);
    this.pitch = Math.asin(Math.max(-1, Math.min(1, f[1])));
    const tx = Math.hypot(c.right[0], c.right[1], c.right[2]);
    const ty = Math.hypot(c.up[0], c.up[1], c.up[2]);
    this.fovDeg = (2 * Math.atan(ty) * 180) / Math.PI;
    void tx;
  }

  forwardVec(): Float32Array {
    const cp = Math.cos(this.pitch);
    return new Float32Array([-Math.sin(this.yaw) * cp, Math.sin(this.pitch), -Math.cos(this.yaw) * cp]);
  }

  update(dt: number) {
    const f = this.forwardVec();
    const r = normalize(cross(f, WORLD_UP));
    const sp = this.speed * (this.keys.has("ShiftLeft") || this.keys.has("ShiftRight") ? 4 : 1) * dt;
    const move = (v: Float32Array, s: number) => { this.eye[0] += v[0] * s; this.eye[1] += v[1] * s; this.eye[2] += v[2] * s; };
    if (this.keys.has("KeyW")) move(f, sp);
    if (this.keys.has("KeyS")) move(f, -sp);
    if (this.keys.has("KeyD")) move(r, sp);
    if (this.keys.has("KeyA")) move(r, -sp);
    if (this.keys.has("KeyE") || this.keys.has("Space")) this.eye[1] += sp;
    if (this.keys.has("KeyQ") || this.keys.has("ControlLeft")) this.eye[1] -= sp;
  }

  state(aspect: number): CameraState {
    const f = this.forwardVec();
    const r = normalize(cross(f, WORLD_UP));
    const u = normalize(cross(r, f));
    const ty = Math.tan((this.fovDeg * Math.PI) / 360);
    const tx = ty * aspect;
    return {
      eye: new Float32Array(this.eye),
      right: new Float32Array([r[0] * tx, r[1] * tx, r[2] * tx]),
      up: new Float32Array([u[0] * ty, u[1] * ty, u[2] * ty]),
      forward: f,
    };
  }
}

/** Exact repo camera -> CameraState (no re-derivation; used by the PSNR harness). */
export function repoCameraState(c: RepoCamera): CameraState {
  const f = normalize(cross(c.up, c.right));
  return { eye: new Float32Array(c.eye), right: new Float32Array(c.right), up: new Float32Array(c.up), forward: f };
}
