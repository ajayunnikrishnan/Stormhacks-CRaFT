/**
 * Map inset (§4): top-down view of the true geometry in the conformal model, y = x̄/(1 + x0):
 * Poincaré ball for H³ (ideal boundary = unit circle), stereographic projection for S³, metres
 * for E³. Draws the fundamental domain's edges as geodesic polylines, the scene footprint, and
 * the viewer with heading. Drawn on its own canvas; click toggles the enlarged view.
 */
import { type Kappa, type V4, type M4, geodesic, tangentToward, toConformal, v4 } from "../geometry/space";
import type { Domain } from "../topology/domain";

export interface MapState {
  kappa: Kappa;
  scale: number;
  W: M4;
  camWorld: V4;
  headingWorld: V4; // unit tangent at camWorld, horizontal forward
  domain: Domain | null;
  sceneHalfModel: [number, number, number];
}

export class MapInset {
  private ctx: CanvasRenderingContext2D;
  enlarged = false;
  constructor(readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    canvas.addEventListener("click", () => this.toggle());
  }
  toggle() {
    this.enlarged = !this.enlarged;
    this.canvas.classList.toggle("big", this.enlarged);
  }

  draw(st: MapState) {
    const { ctx, canvas } = this;
    const css = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const size = Math.round(Math.min(css.width, css.height) * dpr);
    if (canvas.width !== size || canvas.height !== size) { canvas.width = size; canvas.height = size; }
    ctx.clearRect(0, 0, size, size);
    const cx = size / 2, cy = size / 2;
    const unit = st.kappa < 0 ? size * 0.46 : st.kappa > 0 ? size * 0.3 : (size * 0.44) / Math.max(st.sceneHalfModel[0], st.sceneHalfModel[2], st.domain ? st.domain.inradius : 0, 1e-3);
    const toPx = (y: Float64Array): [number, number] => [cx + y[0] * unit, cy + y[2] * unit];
    const conf = (x: V4) => toConformal(x);
    const lw = Math.max(1, size / 220);

    // ideal boundary (H³) / equator (S³)
    if (st.kappa !== 0) {
      ctx.strokeStyle = st.kappa < 0 ? "rgba(120,170,255,0.85)" : "rgba(255,170,110,0.6)";
      ctx.lineWidth = lw * 1.4;
      ctx.beginPath(); ctx.arc(cx, cy, unit, 0, Math.PI * 2); ctx.stroke();
      if (st.kappa < 0) { ctx.fillStyle = "rgba(120,170,255,0.06)"; ctx.fill(); }
    }
    // fundamental domain edges
    if (st.domain) {
      const d = st.domain;
      ctx.strokeStyle = "rgba(255,255,255,0.75)"; ctx.lineWidth = lw;
      const onFace = (f: number, x: V4) => Math.abs(d.faces[f].w[0] * x[0] + d.faces[f].w[1] * x[1] + d.faces[f].w[2] * x[2] + d.faces[f].w[3] * x[3]) < 1e-6;
      const vs = d.vertices;
      for (let i = 0; i < vs.length; i++) for (let j = i + 1; j < vs.length; j++) {
        let common = 0;
        for (let f = 0; f < d.faces.length; f++) if (onFace(f, vs[i]) && onFace(f, vs[j])) common++;
        if (common !== 2) continue;
        this.geodesicPolyline(st.kappa, vs[i], vs[j], (x) => toPx(conf(x)), 12);
      }
    }
    // scene footprint at the eye plane
    ctx.strokeStyle = "rgba(140,255,180,0.7)"; ctx.lineWidth = lw;
    const hx = st.sceneHalfModel[0], hz = st.sceneHalfModel[2];
    const corners = [[-hx, 0, -hz], [hx, 0, -hz], [hx, 0, hz], [-hx, 0, hz]].map((c) => embed(st.kappa, c));
    for (let i = 0; i < 4; i++) this.geodesicPolyline(st.kappa, corners[i], corners[(i + 1) % 4], (x) => toPx(conf(x)), 10);
    // viewer + heading (a short geodesic ahead)
    const pp = toPx(conf(st.camWorld));
    const aheadLen = st.kappa === 0 ? 0.25 * Math.max(hx, hz) : 0.25;
    ctx.strokeStyle = "rgba(255,255,255,0.95)"; ctx.lineWidth = lw * 1.6;
    ctx.beginPath();
    for (let i = 0; i <= 8; i++) { const q = toPx(conf(geodesic(st.kappa, st.camWorld, st.headingWorld, (aheadLen * i) / 8))); if (i === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]); }
    ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(pp[0], pp[1], lw * 3, 0, Math.PI * 2); ctx.fill();
    // label
    ctx.fillStyle = "rgba(230,235,245,0.85)"; ctx.font = `${Math.max(10, size / 20)}px ui-sans-serif, system-ui, sans-serif`; ctx.textAlign = "left";
    ctx.fillText(st.kappa < 0 ? "Poincaré ball · top view" : st.kappa > 0 ? "Stereographic · top view" : "Top view · metres", 10 * lw, 16 * lw);
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
