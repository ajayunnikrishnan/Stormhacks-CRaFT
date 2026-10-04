/**
 * Frame pipeline:
 *   flat   (Phase 1 reference): sv_prepass → walk_flat → composite
 *   curved (Phase 3+):          sv_prepass_curved → walk_curved → composite, any κ, camera at the origin
 * Scene textures are uploaded once; the curved per-κ textures (a_i and (R, cs R)) are
 * re-uploaded when the curvature changes. Pass timings via EXT_disjoint_timer_query_webgl2.
 */
import { FULLSCREEN_VS, FullscreenQuad, Program, TEX_W, dataTexture, destroyTarget, renderTarget, type DataTexture, type RenderTarget } from "./gl";
import { SHADERS, curvedVariant } from "./shaders";
import { startCell, type FoamScene } from "../foam/scene";
import { computeCurvedSites, curvatureParams, sceneCentre, startCellCurved, type CurvedSites } from "../foam/curved";
import type { M4, V4 } from "../geometry/space";

/** Flat camera (repo convention). */
export interface CameraState {
  eye: Float32Array;
  right: Float32Array; // scaled by tan(fovx/2)
  up: Float32Array; // scaled by tan(fovy/2)
  forward: Float32Array; // unit
}

/** Curved camera: camera at the origin, world carried by W. */
export interface IsoCameraState {
  invW: M4; // camera → world
  rayO: V4; // camera position in world model coords (= invW · o)
  tanHalfFov: [number, number];
}

export interface RenderOptions {
  threshold: number;
  nearCull: boolean;
  repoPixelGrid: boolean;
  background: [number, number, number];
}

export interface FrameStats { svMs: number; walkMs: number; width: number; height: number; startCell: number; mode: string }

export class Renderer {
  private quad: FullscreenQuad;
  private progSv: Program;
  private progSvCurvedK: Record<number, Program>;
  private progWalk: Program;
  private progWalkCurvedK: Record<number, Program>;
  private progComposite: Program;
  private siteRgb: RenderTarget;
  private walkTarget: RenderTarget | null = null;
  private timer: { ext: unknown; pending: { q: WebGLQuery; pass: "sv" | "walk" }[] } | null = null;
  stats: FrameStats = { svMs: 0, walkMs: 0, width: 0, height: 0, startCell: 0, mode: "flat" };

  // curved state
  readonly centre: Float64Array;
  curved: CurvedSites;
  private texA: DataTexture;
  private texRad: DataTexture;
  private texAdjOffU: DataTexture;
  private texAdjIdxU: DataTexture;

  constructor(readonly gl: WebGL2RenderingContext, readonly scene: FoamScene) {
    this.quad = new FullscreenQuad(gl);
    this.progSv = new Program(gl, FULLSCREEN_VS, SHADERS.svPrepass, "sv_prepass");
    this.progWalk = new Program(gl, FULLSCREEN_VS, SHADERS.walkFlat, "walk_flat");
    this.progSvCurvedK = {}; this.progWalkCurvedK = {};
    for (const kap of [-1, 0, 1] as const) {
      const v = curvedVariant(kap);
      this.progSvCurvedK[kap] = new Program(gl, FULLSCREEN_VS, v.svPrepassCurved, `sv_prepass_curved[${kap}]`);
      this.progWalkCurvedK[kap] = new Program(gl, FULLSCREEN_VS, v.walkCurved, `walk_curved[${kap}]`);
    }
    this.progComposite = new Program(gl, FULLSCREEN_VS, SHADERS.composite, "composite");
    const rows = Math.max(1, Math.ceil((scene.n * scene.k) / TEX_W));
    this.siteRgb = renderTarget(gl, TEX_W, rows, gl.RGBA16F);
    const ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    if (ext) this.timer = { ext, pending: [] };

    this.centre = sceneCentre(scene);
    this.curved = computeCurvedSites(scene, curvatureParams(0), this.centre);
    this.texA = dataTexture(gl, this.curved.A, scene.n, 4, gl.RGBA32F, gl.RGBA, gl.FLOAT, "a");
    this.texRad = dataTexture(gl, this.curved.rad, scene.n, 2, gl.RG32F, gl.RG, gl.FLOAT, "rad");
    // union adjacency (falls back to the flat graph if the export has no sweep)
    this.texAdjOffU = scene.texAdjOffU ?? scene.texAdjOff;
    this.texAdjIdxU = scene.texAdjIdxU ?? scene.texAdjIdx;
  }

  /** Curvature in 1/m² (sign = κ). Recomputes a_i and radii and re-uploads two textures. */
  setCurvature(k: number) {
    const params = curvatureParams(k);
    if (params.k === this.curved.params.k) return;
    this.curved = computeCurvedSites(this.scene, params, this.centre);
    const gl = this.gl, n = this.scene.n;
    const rows = Math.ceil(n / TEX_W);
    const padA = new Float32Array(TEX_W * rows * 4); padA.set(this.curved.A);
    const padR = new Float32Array(TEX_W * rows * 2); padR.set(this.curved.rad);
    gl.bindTexture(gl.TEXTURE_2D, this.texA.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, rows, gl.RGBA, gl.FLOAT, padA);
    gl.bindTexture(gl.TEXTURE_2D, this.texRad.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, rows, gl.RG, gl.FLOAT, padR);
  }

  // ---------------------------------------------------------------- timers
  private beginTimer(pass: "sv" | "walk") {
    if (!this.timer) return;
    const q = this.gl.createQuery()!;
    this.gl.beginQuery((this.timer.ext as { TIME_ELAPSED_EXT: number }).TIME_ELAPSED_EXT, q);
    this.timer.pending.push({ q, pass });
  }
  private endTimer() { if (this.timer) this.gl.endQuery((this.timer.ext as { TIME_ELAPSED_EXT: number }).TIME_ELAPSED_EXT); }
  private pollTimers() {
    if (!this.timer) return;
    const gl = this.gl;
    const keep: typeof this.timer.pending = [];
    for (const p of this.timer.pending) {
      const avail = gl.getQueryParameter(p.q, gl.QUERY_RESULT_AVAILABLE) as boolean;
      const disjoint = gl.getParameter((this.timer.ext as { GPU_DISJOINT_EXT: number }).GPU_DISJOINT_EXT) as boolean;
      if (avail && !disjoint) {
        const ms = (gl.getQueryParameter(p.q, gl.QUERY_RESULT) as number) / 1e6;
        if (p.pass === "sv") this.stats.svMs = this.stats.svMs * 0.9 + ms * 0.1; else this.stats.walkMs = this.stats.walkMs * 0.9 + ms * 0.1;
        gl.deleteQuery(p.q);
      } else if (avail) gl.deleteQuery(p.q);
      else keep.push(p);
    }
    this.timer.pending = keep;
  }

  private ensureWalkTarget(w: number, h: number, internalFormat: number) {
    const t = this.walkTarget;
    if (t && t.width === w && t.height === h && t.internalFormat === internalFormat) return t;
    if (t) destroyTarget(this.gl, t);
    this.walkTarget = renderTarget(this.gl, w, h, internalFormat, this.gl.LINEAR);
    return this.walkTarget;
  }

  // ---------------------------------------------------------------- flat passes
  private svPrepass(cam: CameraState) {
    const gl = this.gl, s = this.scene, p = this.progSv;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.siteRgb.fbo);
    gl.viewport(0, 0, this.siteRgb.width, this.siteRgb.height);
    p.use();
    p.tex("uPos", 0, s.texPos.tex); p.tex("uSiteOff", 1, s.texSiteOff.tex); p.tex("uSvAxis", 2, s.texSvAxis.tex); p.tex("uSvRgb", 3, s.texSvRgb.tex);
    p.u1i("uK", s.k); p.u1i("uD", s.d); p.u1i("uCount", s.n * s.k);
    p.u3fv("uEye", cam.eye);
    this.beginTimer("sv"); this.quad.draw(); this.endTimer();
  }

  private walk(cam: CameraState, opts: RenderOptions, w: number, h: number, internalFormat: number): RenderTarget {
    const gl = this.gl, s = this.scene, p = this.progWalk;
    const target = this.ensureWalkTarget(w, h, internalFormat);
    const start = startCell(s, cam.eye);
    this.stats.startCell = start; this.stats.mode = "flat";
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    gl.viewport(0, 0, w, h);
    p.use();
    p.tex("uPos", 0, s.texPos.tex); p.tex("uNSigma", 1, s.texNSigma.tex); p.tex("uSiteOff", 2, s.texSiteOff.tex); p.tex("uSiteRgb", 3, this.siteRgb.tex);
    p.tex("uAdjOff", 4, s.texAdjOff.tex); p.tex("uAdjIdx", 5, s.texAdjIdx.tex);
    p.u1i("uK", s.k); p.u1i("uStart", start); p.u1f("uThreshold", opts.threshold);
    p.u1i("uNearCull", opts.nearCull ? 1 : 0); p.u1i("uRepoPixelGrid", opts.repoPixelGrid ? 1 : 0);
    p.u2f("uResolution", w, h);
    p.u3fv("uEye", cam.eye); p.u3fv("uRight", cam.right); p.u3fv("uUp", cam.up); p.u3fv("uForward", cam.forward);
    p.u3f("uBackground", opts.background[0], opts.background[1], opts.background[2]);
    this.beginTimer("walk"); this.quad.draw(); this.endTimer();
    return target;
  }

  // ---------------------------------------------------------------- curved passes
  private svPrepassCurved(cam: IsoCameraState) {
    const gl = this.gl, s = this.scene, c = this.curved, p = this.progSvCurvedK[c.params.kappa];
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.siteRgb.fbo);
    gl.viewport(0, 0, this.siteRgb.width, this.siteRgb.height);
    p.use();
    p.tex("uA", 0, this.texA.tex); p.tex("uRad", 1, this.texRad.tex); p.tex("uSiteOff", 2, s.texSiteOff.tex); p.tex("uSvAxis", 3, s.texSvAxis.tex); p.tex("uSvRgb", 4, s.texSvRgb.tex);
    p.u1i("uK", s.k); p.u1i("uD", s.d); p.u1i("uCount", s.n * s.k);
    p.u1i("uKappa", c.params.kappa); p.u1f("uScale", c.params.scale);
    p.u4fv("uRayO", Float32Array.from(cam.rayO));
    this.beginTimer("sv"); this.quad.draw(); this.endTimer();
  }

  private walkCurved(cam: IsoCameraState, opts: RenderOptions, w: number, h: number, internalFormat: number): RenderTarget {
    const gl = this.gl, s = this.scene, c = this.curved, p = this.progWalkCurvedK[c.params.kappa];
    const target = this.ensureWalkTarget(w, h, internalFormat);
    const start = startCellCurved(c, cam.rayO);
    this.stats.startCell = start; this.stats.mode = `curved κ=${c.params.kappa}`;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    gl.viewport(0, 0, w, h);
    p.use();
    p.tex("uA", 0, this.texA.tex); p.tex("uRad", 1, this.texRad.tex); p.tex("uNSigma", 2, s.texNSigma.tex); p.tex("uSiteOff", 3, s.texSiteOff.tex);
    p.tex("uSiteRgb", 4, this.siteRgb.tex); p.tex("uAdjOff", 5, this.texAdjOffU.tex); p.tex("uAdjIdx", 6, this.texAdjIdxU.tex);
    p.u1i("uKappa", c.params.kappa); p.u1f("uScale", c.params.scale);
    p.u1i("uK", s.k); p.u1i("uStart", start); p.u1f("uThreshold", opts.threshold);
    p.u1i("uNearCull", opts.nearCull ? 1 : 0); p.u1i("uRepoPixelGrid", opts.repoPixelGrid ? 1 : 0);
    p.u1f("uTMax", c.params.kappa > 0 ? 2 * Math.PI : 1e30);
    p.u2f("uResolution", w, h);
    p.u2f("uTanHalfFov", cam.tanHalfFov[0], cam.tanHalfFov[1]);
    p.u4fv("uRayO", Float32Array.from(cam.rayO));
    p.umat4("uInvW", Float32Array.from(cam.invW));
    p.u3f("uBackground", opts.background[0], opts.background[1], opts.background[2]);
    this.beginTimer("walk"); this.quad.draw(); this.endTimer();
    return target;
  }

  // ---------------------------------------------------------------- public
  private composite(target: RenderTarget, canvasW: number, canvasH: number) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasW, canvasH);
    const p = this.progComposite;
    p.use();
    p.tex("uColor", 0, target.tex);
    p.u2f("uResolution", canvasW, canvasH);
    this.quad.draw();
    this.pollTimers();
  }

  frame(cam: CameraState, opts: RenderOptions, canvasW: number, canvasH: number, renderScale: number) {
    const w = Math.max(8, Math.round(canvasW * renderScale)), h = Math.max(8, Math.round(canvasH * renderScale));
    this.stats.width = w; this.stats.height = h;
    this.svPrepass(cam);
    this.composite(this.walk(cam, opts, w, h, this.gl.RGBA16F), canvasW, canvasH);
  }

  frameCurved(cam: IsoCameraState, opts: RenderOptions, canvasW: number, canvasH: number, renderScale: number) {
    const w = Math.max(8, Math.round(canvasW * renderScale)), h = Math.max(8, Math.round(canvasH * renderScale));
    this.stats.width = w; this.stats.height = h;
    this.svPrepassCurved(cam);
    this.composite(this.walkCurved(cam, opts, w, h, this.gl.RGBA16F), canvasW, canvasH);
  }

  private readback(target: RenderTarget, w: number, h: number): Float32Array {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    const buf = new Float32Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const out = new Float32Array(w * h * 3);
    for (let y = 0; y < h; y++) {
      const srcRow = (h - 1 - y) * w;
      for (let x = 0; x < w; x++) { const si = (srcRow + x) * 4, di = (y * w + x) * 3; out[di] = buf[si]; out[di + 1] = buf[si + 1]; out[di + 2] = buf[si + 2]; }
    }
    return out;
  }

  /** Offline flat render: exact size, fp32, RGB rows top-down. */
  renderToArray(cam: CameraState, opts: RenderOptions, w: number, h: number): Float32Array {
    this.svPrepass(cam);
    return this.readback(this.walk(cam, opts, w, h, this.gl.RGBA32F), w, h);
  }
  /** Offline curved render at the current curvature. */
  renderCurvedToArray(cam: IsoCameraState, opts: RenderOptions, w: number, h: number): Float32Array {
    this.svPrepassCurved(cam);
    return this.readback(this.walkCurved(cam, opts, w, h, this.gl.RGBA32F), w, h);
  }

  /** Benchmark: pre-pass + walk `frames` times at w×h, synchronised by a 1-px readback. ms/frame. */
  bench(cam: CameraState | IsoCameraState, opts: RenderOptions, w: number, h: number, frames: number): { msPerFrame: number; svOnlyMs: number } {
    const gl = this.gl;
    const curved = "invW" in cam;
    const px = new Uint16Array(4);
    const sync = (fbo: WebGLFramebuffer) => { gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.HALF_FLOAT, px); };
    const sv = () => (curved ? this.svPrepassCurved(cam as IsoCameraState) : this.svPrepass(cam as CameraState));
    const wk = () => (curved ? this.walkCurved(cam as IsoCameraState, opts, w, h, gl.RGBA16F) : this.walk(cam as CameraState, opts, w, h, gl.RGBA16F));
    sv(); const t = wk(); sync(t.fbo);
    let t0 = performance.now();
    for (let i = 0; i < frames; i++) sv();
    sync(this.siteRgb.fbo);
    const svOnlyMs = (performance.now() - t0) / frames;
    t0 = performance.now();
    for (let i = 0; i < frames; i++) { sv(); wk(); }
    sync(t.fbo);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { msPerFrame: (performance.now() - t0) / frames, svOnlyMs };
  }
}
