/**
 * Frame pipeline (flat, Phase 1):
 *   1. SV pre-pass: per detail site colour for the current eye  -> texSiteRgb (RGBA16F, N*k)
 *   2. walk pass:   per-pixel foam traversal at render scale      -> offscreen RGBA16F
 *   3. composite:   upscale to canvas
 * Timing of each pass is measured with EXT_disjoint_timer_query_webgl2 when available.
 */
import { FULLSCREEN_VS, FullscreenQuad, Program, TEX_W, destroyTarget, renderTarget, type RenderTarget } from "./gl";
import { SHADERS } from "./shaders";
import { startCell, type FoamScene } from "../foam/scene";

export interface CameraState {
  eye: Float32Array; // 3
  right: Float32Array; // 3, scaled by tan(fovx/2)
  up: Float32Array; // 3, scaled by tan(fovy/2)
  forward: Float32Array; // 3, unit
}

export interface RenderOptions {
  threshold: number;
  nearCull: boolean;
  repoPixelGrid: boolean;
  background: [number, number, number];
}

export interface FrameStats { svMs: number; walkMs: number; width: number; height: number; startCell: number }

export class Renderer {
  private quad: FullscreenQuad;
  private progSv: Program;
  private progWalk: Program;
  private progComposite: Program;
  private siteRgb: RenderTarget;
  private walkTarget: RenderTarget | null = null;
  private timer: { ext: unknown; queries: WebGLQuery[]; pending: { q: WebGLQuery; pass: "sv" | "walk" }[] } | null = null;
  stats: FrameStats = { svMs: 0, walkMs: 0, width: 0, height: 0, startCell: 0 };

  constructor(readonly gl: WebGL2RenderingContext, readonly scene: FoamScene) {
    this.quad = new FullscreenQuad(gl);
    this.progSv = new Program(gl, FULLSCREEN_VS, SHADERS.svPrepass, "sv_prepass");
    this.progWalk = new Program(gl, FULLSCREEN_VS, SHADERS.walkFlat, "walk_flat");
    this.progComposite = new Program(gl, FULLSCREEN_VS, SHADERS.composite, "composite");
    const rows = Math.max(1, Math.ceil((scene.n * scene.k) / TEX_W));
    this.siteRgb = renderTarget(gl, TEX_W, rows, gl.RGBA16F);
    const ext = gl.getExtension("EXT_disjoint_timer_query_webgl2");
    if (ext) this.timer = { ext, queries: [], pending: [] };
  }

  private beginTimer(pass: "sv" | "walk") {
    if (!this.timer) return;
    const gl = this.gl;
    const q = gl.createQuery()!;
    gl.beginQuery((this.timer.ext as { TIME_ELAPSED_EXT: number }).TIME_ELAPSED_EXT, q);
    this.timer.pending.push({ q, pass });
  }
  private endTimer() {
    if (!this.timer) return;
    this.gl.endQuery((this.timer.ext as { TIME_ELAPSED_EXT: number }).TIME_ELAPSED_EXT);
  }
  private pollTimers() {
    if (!this.timer) return;
    const gl = this.gl;
    const keep: typeof this.timer.pending = [];
    for (const p of this.timer.pending) {
      const avail = gl.getQueryParameter(p.q, gl.QUERY_RESULT_AVAILABLE) as boolean;
      const disjoint = gl.getParameter((this.timer.ext as { GPU_DISJOINT_EXT: number }).GPU_DISJOINT_EXT) as boolean;
      if (avail && !disjoint) {
        const ns = gl.getQueryParameter(p.q, gl.QUERY_RESULT) as number;
        const ms = ns / 1e6;
        if (p.pass === "sv") this.stats.svMs = this.stats.svMs * 0.9 + ms * 0.1;
        else this.stats.walkMs = this.stats.walkMs * 0.9 + ms * 0.1;
        gl.deleteQuery(p.q);
      } else if (avail) {
        gl.deleteQuery(p.q);
      } else keep.push(p);
    }
    this.timer.pending = keep;
  }

  /** Pass 1: per-site SV colours for this eye position. */
  private svPrepass(cam: CameraState) {
    const gl = this.gl, s = this.scene, p = this.progSv;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.siteRgb.fbo);
    gl.viewport(0, 0, this.siteRgb.width, this.siteRgb.height);
    p.use();
    p.tex("uPos", 0, s.texPos.tex);
    p.tex("uSiteOff", 1, s.texSiteOff.tex);
    p.tex("uSvAxis", 2, s.texSvAxis.tex);
    p.tex("uSvRgb", 3, s.texSvRgb.tex);
    p.u1i("uK", s.k);
    p.u1i("uD", s.d);
    p.u1i("uCount", s.n * s.k);
    p.u3fv("uEye", cam.eye);
    this.beginTimer("sv");
    this.quad.draw();
    this.endTimer();
  }

  private ensureWalkTarget(w: number, h: number, internalFormat: number) {
    const gl = this.gl;
    const t = this.walkTarget;
    if (t && t.width === w && t.height === h && t.internalFormat === internalFormat) return t;
    if (t) destroyTarget(gl, t);
    this.walkTarget = renderTarget(gl, w, h, internalFormat, gl.LINEAR);
    return this.walkTarget;
  }

  /** Pass 2: the foam walk into an offscreen target of the given size. */
  private walk(cam: CameraState, opts: RenderOptions, w: number, h: number, internalFormat: number): RenderTarget {
    const gl = this.gl, s = this.scene, p = this.progWalk;
    const target = this.ensureWalkTarget(w, h, internalFormat);
    const start = startCell(s, cam.eye);
    this.stats.startCell = start;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    gl.viewport(0, 0, w, h);
    p.use();
    p.tex("uPos", 0, s.texPos.tex);
    p.tex("uNSigma", 1, s.texNSigma.tex);
    p.tex("uSiteOff", 2, s.texSiteOff.tex);
    p.tex("uSiteRgb", 3, this.siteRgb.tex);
    p.tex("uAdjOff", 4, s.texAdjOff.tex);
    p.tex("uAdjIdx", 5, s.texAdjIdx.tex);
    p.u1i("uK", s.k);
    p.u1i("uStart", start);
    p.u1f("uThreshold", opts.threshold);
    p.u1i("uNearCull", opts.nearCull ? 1 : 0);
    p.u1i("uRepoPixelGrid", opts.repoPixelGrid ? 1 : 0);
    p.u2f("uResolution", w, h);
    p.u3fv("uEye", cam.eye);
    p.u3fv("uRight", cam.right);
    p.u3fv("uUp", cam.up);
    p.u3fv("uForward", cam.forward);
    p.u3f("uBackground", opts.background[0], opts.background[1], opts.background[2]);
    this.beginTimer("walk");
    this.quad.draw();
    this.endTimer();
    return target;
  }

  /** Full frame to the canvas. renderScale in (0,1]. */
  frame(cam: CameraState, opts: RenderOptions, canvasW: number, canvasH: number, renderScale: number) {
    const gl = this.gl;
    const w = Math.max(8, Math.round(canvasW * renderScale));
    const h = Math.max(8, Math.round(canvasH * renderScale));
    this.stats.width = w;
    this.stats.height = h;
    this.svPrepass(cam);
    const target = this.walk(cam, opts, w, h, gl.RGBA16F);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, canvasW, canvasH);
    const p = this.progComposite;
    p.use();
    p.tex("uColor", 0, target.tex);
    p.u2f("uResolution", canvasW, canvasH);
    this.quad.draw();
    this.pollTimers();
  }

  /** Benchmark: run pre-pass + walk `frames` times at w×h and block on gl.finish(). Returns ms/frame. */
  bench(cam: CameraState, opts: RenderOptions, w: number, h: number, frames: number): { msPerFrame: number; svOnlyMs: number } {
    const gl = this.gl;
    // gl.finish() does not block reliably (e.g. hidden tab); a 1-pixel readback does.
    const px = new Uint16Array(4);
    const sync = (fbo: WebGLFramebuffer) => { gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.HALF_FLOAT, px); };
    this.svPrepass(cam); const t = this.walk(cam, opts, w, h, gl.RGBA16F); sync(t.fbo); // warm up
    let t0 = performance.now();
    for (let i = 0; i < frames; i++) this.svPrepass(cam);
    sync(this.siteRgb.fbo);
    const svOnlyMs = (performance.now() - t0) / frames;
    t0 = performance.now();
    for (let i = 0; i < frames; i++) { this.svPrepass(cam); this.walk(cam, opts, w, h, gl.RGBA16F); }
    sync(t.fbo);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { msPerFrame: (performance.now() - t0) / frames, svOnlyMs };
  }

  /**
   * Offline render for tests: exact size, fp32 target, returns RGB rows top-down
   * (matching the reference renderer's image layout).
   */
  renderToArray(cam: CameraState, opts: RenderOptions, w: number, h: number): Float32Array {
    const gl = this.gl;
    this.svPrepass(cam);
    const target = this.walk(cam, opts, w, h, gl.RGBA32F);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    const buf = new Float32Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const out = new Float32Array(w * h * 3);
    for (let y = 0; y < h; y++) {
      const srcRow = (h - 1 - y) * w; // flip: GL rows are bottom-up
      for (let x = 0; x < w; x++) {
        const si = (srcRow + x) * 4, di = (y * w + x) * 3;
        out[di] = buf[si]; out[di + 1] = buf[si + 1]; out[di + 2] = buf[si + 2];
      }
    }
    return out;
  }
}
