/**
 * 2D overlay (canvas on top of the GL view): compass, map inset, laser geodesic.
 *
 * Map inset (§4 Tools): orthographic top-down view of the conformal model, y = x̄/(1 + x0):
 * Poincaré ball for H³ (ideal boundary = unit circle), stereographic projection for S³,
 * plain metres for E³. Draws the domain's edges as geodesic polylines, the scene footprint,
 * the player and heading, beacons and the laser.
 *
 * Laser: the camera-ray geodesic is drawn as an image-space polyline: a world point y seen
 * from the camera at o appears in the direction of the tangent toward y (Eq. 5), so we
 * project sample points of the geodesic through the camera. (Overlay only — not occluded.)
 */
import { type Kappa, type V4, type M4, apply, geodesic, tangentToward, toConformal, v4, geodesicDir } from "../geometry/space";
import type { Domain } from "../topology/domain";

export interface OverlayState {
  kappa: Kappa;
  scale: number;
  W: M4; // world → camera
  invW: M4;
  camWorld: V4;
  headingWorld: V4; // unit tangent at camWorld, horizontal forward
  compassAngle: number; // radians, + = right of heading
  domain: Domain | null;
  sceneHalfModel: [number, number, number]; // scene bbox half-extents in model units
  tanHalfFov: [number, number];
  laser: { o: V4; v: V4; length: number } | null; // world coords
  beacons: V4[];
}

export class Overlay {
  private ctx: CanvasRenderingContext2D;
  showMap = true;
  showCompass = true;
  constructor(readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
  }

  resize(w: number, h: number) {
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
  }

  draw(st: OverlayState) {
    const { ctx } = this;
    const W = this.canvas.width, H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);
    if (st.laser) this.drawLaser(st, W, H);
    if (this.showCompass) this.drawCompass(st, W, H);
    if (this.showMap) this.drawMap(st, W, H);
  }

  // ---------------------------------------------------------------- projection helpers
  /** Project a world point to pixel coords (or null if behind the camera). */
  project(st: OverlayState, y: V4, W: number, H: number): [number, number] | null {
    const { u } = tangentToward(st.kappa, st.camWorld, y);
    const c = apply(st.W, u); // camera-space tangent (camera at origin)
    const dz = -c[3];
    if (dz <= 1e-6) return null;
    const px = (c[1] / dz) / st.tanHalfFov[0], py = (c[2] / dz) / st.tanHalfFov[1];
    return [(px * 0.5 + 0.5) * W, (0.5 - py * 0.5) * H];
  }

  private drawLaser(st: OverlayState, W: number, H: number) {
    const { ctx } = this;
    const L = st.laser!;
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255,60,60,0.9)";
    ctx.beginPath();
    let pen = false;
    const n = 96;
    for (let i = 1; i <= n; i++) {
      const t = (L.length * i) / n;
      const p = this.project(st, geodesic(st.kappa, L.o, L.v, t), W, H);
      if (!p) { pen = false; continue; }
      if (!pen) { ctx.moveTo(p[0], p[1]); pen = true; } else ctx.lineTo(p[0], p[1]);
    }
    ctx.stroke();
  }

  private drawCompass(st: OverlayState, W: number, H: number) {
    const { ctx } = this;
    const cx = W - 70, cy = H - 70, r = 44;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.strokeStyle = "rgba(255,255,255,0.6)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(0, 0, r, 0, Math.PI * 2); ctx.stroke();
    // heading tick (up)
    ctx.beginPath(); ctx.moveTo(0, -r); ctx.lineTo(0, -r + 8); ctx.stroke();
    // compass needle: angle measured from heading, + = right (clockwise on screen)
    const a = st.compassAngle;
    ctx.rotate(a);
    ctx.fillStyle = "rgba(255,200,80,0.95)";
    ctx.beginPath(); ctx.moveTo(0, -r + 4); ctx.lineTo(6, 6); ctx.lineTo(-6, 6); ctx.closePath(); ctx.fill();
    ctx.restore();
    ctx.fillStyle = "#ddd"; ctx.font = "11px ui-monospace, monospace"; ctx.textAlign = "center";
    ctx.fillText(`compass ${((a * 180) / Math.PI).toFixed(1)}°`, cx, cy + r + 14);
  }

  private drawMap(st: OverlayState, W: number, H: number) {
    const { ctx } = this;
    const size = Math.min(220, Math.floor(W * 0.22));
    const x0 = 12, y0 = H - size - 12, cx = x0 + size / 2, cy = y0 + size / 2;
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,0.55)";
    ctx.fillRect(x0, y0, size, size);
    ctx.beginPath(); ctx.rect(x0, y0, size, size); ctx.clip();
    // scale: H³ → unit disc fits; S³ → stereographic, show radius 1.6; E³ → metres-based
    let unit: number;
    if (st.kappa < 0) unit = size * 0.47;
    else if (st.kappa > 0) unit = size * 0.3;
    else unit = (size * 0.45) / Math.max(st.sceneHalfModel[0], st.sceneHalfModel[2], st.domain ? st.domain.inradius : 0, 1e-3);
    const toPx = (y: Float64Array): [number, number] => [cx + y[0] * unit, cy + y[2] * unit];
    const conf = (x: V4) => toConformal(x);
    // ideal boundary / equator
    if (st.kappa !== 0) {
      ctx.strokeStyle = st.kappa < 0 ? "rgba(120,180,255,0.9)" : "rgba(255,180,120,0.6)";
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(cx, cy, unit, 0, Math.PI * 2); ctx.stroke();
    }
    // domain edges: vertices pairs on two common faces
    if (st.domain) {
      const d = st.domain;
      ctx.strokeStyle = "rgba(255,255,255,0.75)"; ctx.lineWidth = 1;
      const onFace = (f: number, x: V4) => Math.abs(d.faces[f].w[0] * x[0] + d.faces[f].w[1] * x[1] + d.faces[f].w[2] * x[2] + d.faces[f].w[3] * x[3]) < 1e-6;
      const vs = d.vertices;
      for (let i = 0; i < vs.length; i++) for (let j = i + 1; j < vs.length; j++) {
        let common = 0;
        for (let f = 0; f < d.faces.length; f++) if (onFace(f, vs[i]) && onFace(f, vs[j])) common++;
        if (common !== 2) continue;
        this.geodesicPolyline(st.kappa, vs[i], vs[j], (x) => toPx(conf(x)), 12);
      }
    }
    // scene footprint (horizontal rectangle at the eye plane)
    ctx.strokeStyle = "rgba(120,255,160,0.7)"; ctx.lineWidth = 1;
    const hx = st.sceneHalfModel[0], hz = st.sceneHalfModel[2];
    const corners = [[-hx, 0, -hz], [hx, 0, -hz], [hx, 0, hz], [-hx, 0, hz]].map((c) => embed(st.kappa, c));
    for (let i = 0; i < 4; i++) this.geodesicPolyline(st.kappa, corners[i], corners[(i + 1) % 4], (x) => toPx(conf(x)), 10);
    // laser
    if (st.laser) {
      ctx.strokeStyle = "rgba(255,80,80,0.9)"; ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i <= 64; i++) {
        const p = toPx(conf(geodesic(st.kappa, st.laser.o, st.laser.v, (st.laser.length * i) / 64)));
        if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
      }
      ctx.stroke();
    }
    // beacons
    ctx.fillStyle = "rgba(255,220,80,0.95)";
    for (const b of st.beacons) { const p = toPx(conf(b)); ctx.beginPath(); ctx.arc(p[0], p[1], 3, 0, Math.PI * 2); ctx.fill(); }
    // player + heading
    const pp = toPx(conf(st.camWorld));
    const ahead = toPx(conf(geodesic(st.kappa, st.camWorld, st.headingWorld, 0.25 * (st.kappa === 0 ? Math.max(hx, hz) : 1))));
    ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(pp[0], pp[1]); ctx.lineTo(ahead[0], ahead[1]); ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(pp[0], pp[1], 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
    ctx.fillStyle = "#ddd"; ctx.font = "11px ui-monospace, monospace"; ctx.textAlign = "left";
    ctx.fillText(st.kappa < 0 ? "map: Poincaré ball (top view)" : st.kappa > 0 ? "map: stereographic (top view)" : "map: metres (top view)", x0 + 4, y0 + 12);
  }

  private geodesicPolyline(k: Kappa, a: V4, b: V4, toPx: (x: V4) => [number, number], n: number) {
    const { ctx } = this;
    const { u, d } = tangentToward(k, a, b);
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
      const p = toPx(geodesic(k, a, u, (d * i) / n));
      if (i === 0) ctx.moveTo(p[0], p[1]); else ctx.lineTo(p[0], p[1]);
    }
    ctx.stroke();
  }
}

function embed(k: Kappa, x: number[]): V4 {
  const n = Math.hypot(x[0], x[1], x[2]);
  if (k === 0) return v4(1, x[0], x[1], x[2]);
  if (n < 1e-12) return v4(1, 0, 0, 0);
  const c = k > 0 ? Math.cos(n) : Math.cosh(n), s = (k > 0 ? Math.sin(n) : Math.sinh(n)) / n;
  return v4(c, s * x[0], s * x[1], s * x[2]);
}
export const _u = { geodesicDir };
