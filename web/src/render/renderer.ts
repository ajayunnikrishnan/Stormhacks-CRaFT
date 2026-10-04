/**
 * Frame pipeline:
 *   flat   (Phase 1 reference): sv_prepass → walk_flat → composite
 *   curved (Phase 3+):          sv_prepass_curved → walk_curved → composite, any κ, camera at the origin
 * Scene textures are uploaded once; the curved per-κ textures (a_i and (R, cs R)) are
 * re-uploaded when the curvature changes. Pass timings via EXT_disjoint_timer_query_webgl2.
 */
import { FULLSCREEN_VS, FullscreenQuad, Program, TEX_W, dataTexture, destroyTarget, renderTarget, mrtTarget, destroyMrt, type DataTexture, type RenderTarget, type MrtTarget } from "./gl";
import type { Lights } from "./lights";
import { SHADERS, curvedVariant } from "./shaders";
import { startCell, type FoamScene } from "../foam/scene";
import { computeCurvedSites, curvatureParams, sceneCentre, startCellCurved, type CurvedSites } from "../foam/curved";
import type { M4, V4 } from "../geometry/space";
import type { Domain } from "../topology/domain";
import { buildLocateGrid, LOC_GRID, type LocateGrid } from "../topology/locate";

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

export interface LightingOptions {
  enabled: boolean;
  lights: Lights | null;
  flashlight: boolean;
  flashColor: [number, number, number];
  flashConeDeg: [number, number]; // inner, outer half-angles
  ambient: number;
  rho: number;
  fogSigmaPerM: number;
  fogColor: [number, number, number];
  exposure: number;
  lampRadiusM: number;
  shadowScale: number; // shadow pass resolution relative to the G-buffer
  camFwdWorld: V4;
}

export const DEFAULT_LIGHTING: LightingOptions = {
  enabled: true, lights: null, flashlight: true, flashColor: [14, 13, 11], flashConeDeg: [14, 24], ambient: 0.25, rho: 1.0,
  fogSigmaPerM: 0.04, fogColor: [0.02, 0.025, 0.035], exposure: 1.2, lampRadiusM: 0.12, shadowScale: 0.5, camFwdWorld: new Float64Array([0, 0, 0, -1]),
};

export interface FrameStats { svMs: number; walkMs: number; width: number; height: number; startCell: number; mode: string; litMs?: number }

export class Renderer {
  private quad: FullscreenQuad;
  private progSv: Program;
  private progSvCurvedK: Record<number, Program>;
  private progWalk: Program;
  private progWalkCurvedK: Record<number, Program>;
  private progShadowK: Record<number, Program>;
  private progShadeK: Record<number, Program>;
  private progComposite: Program;
  private gbuf: MrtTarget | null = null;
  private shadowRt: RenderTarget | null = null;
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
  // topology
  domain: Domain | null = null;
  private locate: LocateGrid | null = null;
  private texLoc: DataTexture;
  maxHops = 16;
  /** in-world tint of light passing through each face pair (gallery colours); 0 disables */
  tintStrength = 0.35;
  static readonly PAIR_COLOURS: [number, number, number][] = [[1.0, 0.55, 0.45], [0.5, 0.85, 1.0], [0.7, 1.0, 0.55], [1.0, 0.85, 0.45], [0.85, 0.6, 1.0], [0.55, 1.0, 0.9]];
  /** set by the app so bench() can time the full lit pipeline */
  benchLighting: LightingOptions | null = null;
  /** Fog / step cutoff distance in metres (§3.9: fog also acts as the distance cutoff). */
  fogDistanceM = 40;

  constructor(readonly gl: WebGL2RenderingContext, readonly scene: FoamScene) {
    this.quad = new FullscreenQuad(gl);
    this.progSv = new Program(gl, FULLSCREEN_VS, SHADERS.svPrepass, "sv_prepass");
    this.progWalk = new Program(gl, FULLSCREEN_VS, SHADERS.walkFlat, "walk_flat");
    this.progSvCurvedK = {}; this.progWalkCurvedK = {}; this.progShadowK = {}; this.progShadeK = {};
    for (const kap of [-1, 0, 1] as const) {
      const v = curvedVariant(kap);
      this.progSvCurvedK[kap] = new Program(gl, FULLSCREEN_VS, v.svPrepassCurved, `sv_prepass_curved[${kap}]`);
      this.progWalkCurvedK[kap] = new Program(gl, FULLSCREEN_VS, v.walkCurved, `walk_curved[${kap}]`);
      this.progShadowK[kap] = new Program(gl, FULLSCREEN_VS, v.shadow, `shadow[${kap}]`);
      this.progShadeK[kap] = new Program(gl, FULLSCREEN_VS, v.shade, `shade[${kap}]`);
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
    this.texLoc = dataTexture(gl, new Uint32Array(12 * LOC_GRID * LOC_GRID), 12 * LOC_GRID * LOC_GRID, 1, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, "locgrid");
  }

  /** Set (or clear) the fundamental domain; rebuilds the point-location grid for the current κ. */
  setDomain(d: Domain | null) {
    this.domain = d;
    this.locate = null;
    if (d) this.rebuildLocateGrid();
  }

  private rebuildLocateGrid() {
    const d = this.domain;
    if (!d) return;
    const adjOff = this.scene.adjOffU ?? this.scene.adjOff, adjIdx = this.scene.adjIdxU ?? this.scene.adjIdx;
    this.locate = buildLocateGrid(d, this.curved, adjOff, adjIdx, this.locate && this.locate.ids.length === d.faces.length * LOC_GRID * LOC_GRID ? this.locate : undefined);
    const gl = this.gl;
    const pad = new Uint32Array(TEX_W * Math.ceil((12 * LOC_GRID * LOC_GRID) / TEX_W));
    pad.set(this.locate.ids);
    gl.bindTexture(gl.TEXTURE_2D, this.texLoc.tex);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_W, pad.length / TEX_W, gl.RED_INTEGER, gl.UNSIGNED_INT, pad);
  }

  private bindDomain(p: Program) {
    const d = this.domain;
    const gl = this.gl;
    if (!d || !this.locate) { p.u1i("uFaceCount", 0); p.u1i("uMaxHops", 0); p.u1f("uTintStrength", 0); p.tex("uLocGrid", 7, this.texLoc.tex); return; }
    const n = d.faces.length;
    const W = new Float32Array(12 * 4), G = new Float32Array(12 * 16), L = new Float32Array(12 * 16), partner = new Int32Array(12), half = new Float32Array(12);
    for (let f = 0; f < n; f++) {
      W.set(d.faces[f].w, f * 4);
      G.set(d.faces[f].g, f * 16);
      const pf = d.faces[f].partner;
      L.set(d.faces[pf].invFrame, f * 16);
      partner[f] = pf;
      half[f] = this.locate.chartHalf[pf];
    }
    gl.uniform4fv(p.loc("uFaceW"), W);
    gl.uniformMatrix4fv(p.loc("uFaceG"), false, G);
    gl.uniformMatrix4fv(p.loc("uFaceLocInv"), false, L);
    gl.uniform1iv(p.loc("uFacePartner"), partner);
    gl.uniform1fv(p.loc("uFaceChartHalf"), half);
    const tint = new Float32Array(12 * 3);
    for (let f = 0; f < n; f++) tint.set(Renderer.PAIR_COLOURS[Math.floor(f / 2) % Renderer.PAIR_COLOURS.length], f * 3);
    gl.uniform3fv(p.loc("uFaceTint"), tint);
    p.u1f("uTintStrength", this.tintStrength);
    p.u1i("uFaceCount", n);
    p.u1i("uMaxHops", this.maxHops);
    p.tex("uLocGrid", 7, this.texLoc.tex);
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
    if (this.domain) this.rebuildLocateGrid();
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

  private ensureGbuf(w: number, h: number): MrtTarget {
    const gl = this.gl;
    const g = this.gbuf;
    if (g && g.width === w && g.height === h) return g;
    if (g) destroyMrt(gl, g);
    this.gbuf = mrtTarget(gl, w, h, [gl.RGBA32F, gl.RGBA32F, gl.RGBA32F, gl.RGBA32F]);
    return this.gbuf;
  }

  /** Primary curved walk into the G-buffer; returns it (attachment 0 = baked colour + transmittance). */
  private walkCurved(cam: IsoCameraState, opts: RenderOptions, w: number, h: number, _internalFormat: number): MrtTarget {
    const gl = this.gl, s = this.scene, c = this.curved, p = this.progWalkCurvedK[c.params.kappa];
    const target = this.ensureGbuf(w, h);
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
    p.u1f("uTMax", Math.min(c.params.kappa > 0 ? 2 * Math.PI : 1e30, this.fogDistanceM * c.params.scale));
    p.u2f("uResolution", w, h);
    p.u2f("uTanHalfFov", cam.tanHalfFov[0], cam.tanHalfFov[1]);
    p.u4fv("uRayO", Float32Array.from(cam.rayO));
    p.umat4("uInvW", Float32Array.from(cam.invW));
    p.u3f("uBackground", opts.background[0], opts.background[1], opts.background[2]);
    this.bindDomain(p);
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

  private lightArrays(lo: LightingOptions) {
    const n = Math.min(8, lo.lights?.list.length ?? 0);
    const pos = new Float32Array(8 * 4), col = new Float32Array(8 * 3);
    for (let i = 0; i < n; i++) { pos.set(lo.lights!.list[i].pos, i * 4); col.set(lo.lights!.list[i].color, i * 3); }
    return { n, pos, col };
  }

  /** Shadow pass: transmittance from the G-buffer hit toward lights 0..3, at reduced resolution. */
  private shadowPass(g: MrtTarget, lo: LightingOptions): RenderTarget {
    const gl = this.gl, s = this.scene, c = this.curved, p = this.progShadowK[c.params.kappa];
    const w = Math.max(4, Math.round(g.width * lo.shadowScale)), h = Math.max(4, Math.round(g.height * lo.shadowScale));
    if (!this.shadowRt || this.shadowRt.width !== w || this.shadowRt.height !== h) {
      if (this.shadowRt) destroyTarget(gl, this.shadowRt);
      this.shadowRt = renderTarget(gl, w, h, gl.RGBA16F, gl.LINEAR);
    }
    const { n, pos } = this.lightArrays(lo);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowRt.fbo);
    gl.viewport(0, 0, w, h);
    p.use();
    p.tex("uA", 0, this.texA.tex); p.tex("uRad", 1, this.texRad.tex); p.tex("uNSigma", 2, s.texNSigma.tex);
    p.tex("uAdjOff", 3, this.texAdjOffU.tex); p.tex("uAdjIdx", 4, this.texAdjIdxU.tex);
    p.tex("uGHit", 5, g.texs[1]); p.tex("uGMeta", 6, g.texs[3]);
    p.u1i("uKappa", c.params.kappa); p.u1f("uScale", c.params.scale);
    p.u1i("uLightCount", Math.min(4, n));
    gl.uniform4fv(p.loc("uLightPos"), pos.subarray(0, 16));
    p.u2f("uGRes", g.width, g.height); p.u2f("uRes", w, h);
    this.quad.draw();
    return this.shadowRt;
  }

  /** Deferred shading + fog + tone map to the given framebuffer (null = canvas). */
  private shadePass(g: MrtTarget, shadow: RenderTarget, cam: IsoCameraState, lo: LightingOptions, fbo: WebGLFramebuffer | null, outW: number, outH: number) {
    const gl = this.gl, c = this.curved, p = this.progShadeK[c.params.kappa];
    const { n, pos, col } = this.lightArrays(lo);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.viewport(0, 0, outW, outH);
    p.use();
    p.tex("uGColor", 0, g.texs[0]); p.tex("uGHit", 1, g.texs[1]); p.tex("uGNormal", 2, g.texs[2]); p.tex("uGMeta", 3, g.texs[3]); p.tex("uShadow", 4, shadow.tex);
    p.u2f("uRes", outW, outH); p.u2f("uGRes", g.width, g.height);
    p.u1i("uKappa", c.params.kappa); p.u1f("uScale", c.params.scale);
    p.u1i("uLightCount", n);
    gl.uniform4fv(p.loc("uLightPos"), pos); gl.uniform3fv(p.loc("uLightColor"), col);
    p.u1f("uLightRadius", lo.lampRadiusM * c.params.scale);
    p.u1i("uFlashOn", lo.flashlight ? 1 : 0);
    p.u4fv("uCamPos", Float32Array.from(cam.rayO)); p.u4fv("uCamFwd", Float32Array.from(lo.camFwdWorld));
    p.u3f("uFlashColor", lo.flashColor[0], lo.flashColor[1], lo.flashColor[2]);
    p.u2f("uFlashCone", Math.cos((lo.flashConeDeg[0] * Math.PI) / 180), Math.cos((lo.flashConeDeg[1] * Math.PI) / 180));
    p.u1f("uAmbient", lo.ambient); p.u1f("uRho", lo.rho);
    p.u1f("uFogSigma", lo.fogSigmaPerM / c.params.scale); p.u3f("uFogColor", lo.fogColor[0], lo.fogColor[1], lo.fogColor[2]);
    p.u1f("uFogEnd", Math.min(c.params.kappa > 0 ? 2 * Math.PI : 1e30, this.fogDistanceM * c.params.scale));
    p.u1f("uExposure", lo.exposure); p.u1i("uLightingOn", lo.enabled ? 1 : 0);
    p.u2f("uTanHalfFov", cam.tanHalfFov[0], cam.tanHalfFov[1]);
    p.umat4("uInvW", Float32Array.from(cam.invW));
    this.quad.draw();
    this.pollTimers();
  }

  frameCurved(cam: IsoCameraState, opts: RenderOptions, canvasW: number, canvasH: number, renderScale: number, lo: LightingOptions = DEFAULT_LIGHTING) {
    const w = Math.max(8, Math.round(canvasW * renderScale)), h = Math.max(8, Math.round(canvasH * renderScale));
    this.stats.width = w; this.stats.height = h;
    // the fog is the walk's distance cutoff: stop where e^{−σt} < 1%
    this.fogDistanceM = lo.enabled && lo.fogSigmaPerM > 0 ? Math.min(this.fogDistanceM, 4.6 / lo.fogSigmaPerM) : this.fogDistanceM;
    this.svPrepassCurved(cam);
    const g = this.walkCurved(cam, opts, w, h, this.gl.RGBA16F);
    const sh = this.shadowPass(g, lo);
    this.shadePass(g, sh, cam, lo, null, canvasW, canvasH);
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
  private readbackFbo(fbo: WebGLFramebuffer, w: number, h: number, attachment = 0): Float32Array {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.readBuffer(gl.COLOR_ATTACHMENT0 + attachment);
    const buf = new Float32Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, buf);
    gl.readBuffer(gl.COLOR_ATTACHMENT0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const out = new Float32Array(w * h * 3);
    for (let y = 0; y < h; y++) {
      const srcRow = (h - 1 - y) * w;
      for (let x = 0; x < w; x++) { const si = (srcRow + x) * 4, di = (y * w + x) * 3; out[di] = buf[si]; out[di + 1] = buf[si + 1]; out[di + 2] = buf[si + 2]; }
    }
    return out;
  }

  /** Offline curved render at the current curvature: BAKED colour (G-buffer attachment 0), for parity tests. */
  renderCurvedToArray(cam: IsoCameraState, opts: RenderOptions, w: number, h: number): Float32Array {
    this.svPrepassCurved(cam);
    const g = this.walkCurved(cam, opts, w, h, this.gl.RGBA32F);
    return this.readbackFbo(g.fbo, w, h, 0);
  }

  /** Offline LIT render (full pipeline) at exact size. */
  renderLitToArray(cam: IsoCameraState, opts: RenderOptions, lo: LightingOptions, w: number, h: number): Float32Array {
    this.svPrepassCurved(cam);
    const g = this.walkCurved(cam, opts, w, h, this.gl.RGBA32F);
    const sh = this.shadowPass(g, lo);
    const out = renderTarget(this.gl, w, h, this.gl.RGBA32F);
    this.shadePass(g, sh, cam, lo, out.fbo, w, h);
    const arr = this.readbackFbo(out.fbo, w, h, 0);
    destroyTarget(this.gl, out);
    return arr;
  }

  /** Benchmark: pre-pass + walk `frames` times at w×h, synchronised by a 1-px readback. ms/frame. */
  bench(cam: CameraState | IsoCameraState, opts: RenderOptions, w: number, h: number, frames: number): { msPerFrame: number; svOnlyMs: number } {
    const gl = this.gl;
    const curved = "invW" in cam;
    const px = new Uint16Array(4), pxf = new Float32Array(4);
    // RGBA16F targets read back as HALF_FLOAT, the RGBA32F G-buffer as FLOAT
    const sync = (fbo: WebGLFramebuffer, float32 = false) => { gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); if (float32) gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, pxf); else gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.HALF_FLOAT, px); };
    const sv = () => (curved ? this.svPrepassCurved(cam as IsoCameraState) : this.svPrepass(cam as CameraState));
    const wk = () => (curved ? this.walkCurved(cam as IsoCameraState, opts, w, h, gl.RGBA16F) : this.walk(cam as CameraState, opts, w, h, gl.RGBA16F));
    sv(); const t = wk(); sync(t.fbo, curved);
    // for the curved path also measure the lighting passes
    if (curved && this.benchLighting) {
      const lo = this.benchLighting;
      let t1 = performance.now();
      for (let i = 0; i < frames; i++) { sv(); const g = this.walkCurved(cam as IsoCameraState, opts, w, h, gl.RGBA16F); const shd = this.shadowPass(g, lo); this.shadePass(g, shd, cam as IsoCameraState, lo, null, w, h); }
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.shadowRt!.fbo); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.HALF_FLOAT, px);
      this.stats.litMs = (performance.now() - t1) / frames;
    }
    let t0 = performance.now();
    for (let i = 0; i < frames; i++) sv();
    sync(this.siteRgb.fbo);
    const svOnlyMs = (performance.now() - t0) / frames;
    t0 = performance.now();
    for (let i = 0; i < frames; i++) { sv(); wk(); }
    sync(t.fbo, curved);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return { msPerFrame: (performance.now() - t0) / frames, svOnlyMs };
  }
}
