/**
 * Loads an exported foam scene (tools/export_scene.py, format "surveyor-foam-v1")
 * and packs it into WebGL2 data textures. See docs/ARCHITECTURE.md §9 for the layout.
 */
import { dataTexture, type DataTexture } from "../render/gl";

export interface Section { offset: number; dtype: string; shape: number[]; nbytes: number }
export interface RepoCamera { name: string; eye: number[]; right: number[]; up: number[]; width: number; height: number }
export interface SceneManifest {
  format: string;
  n: number;
  k: number;
  d: number;
  n_edges_directed: number;
  bbox_min: number[];
  bbox_max: number[];
  bin: string;
  bin_bytes: number;
  sections: Record<string, Section>;
  info: Record<string, unknown>;
  cameras?: RepoCamera[];
}

export interface FoamScene {
  manifest: SceneManifest;
  n: number;
  k: number;
  d: number;
  /** CPU copies needed every frame (start-cell search). */
  pos: Float32Array; // [N,4] xyz, r
  /** GPU textures. */
  texPos: DataTexture; // RGBA32F  N
  texNSigma: DataTexture; // RGBA16F  N
  texSiteOff: DataTexture; // RGBA16F  N*k   (offset xyz, height)
  texSvAxis: DataTexture; // RGB16F   N*k*D
  texSvRgb: DataTexture; // RGB16F   N*k*D
  texAdjOff: DataTexture; // R32UI    N+1
  texAdjIdx: DataTexture; // R32UI    E
}

function view(buf: ArrayBuffer, s: Section): ArrayBufferView {
  const count = s.shape.reduce((a, b) => a * b, 1);
  switch (s.dtype) {
    case "float32": return new Float32Array(buf, s.offset, count);
    case "float16": return new Uint16Array(buf, s.offset, count); // raw half bits
    case "uint32": return new Uint32Array(buf, s.offset, count);
    default: throw new Error(`unsupported dtype ${s.dtype}`);
  }
}

export async function loadScene(gl: WebGL2RenderingContext, url: string, onProgress?: (msg: string) => void): Promise<FoamScene> {
  onProgress?.("fetching manifest");
  const base = url.replace(/\/[^/]*$/, "/");
  const manifest = (await (await fetch(url)).json()) as SceneManifest;
  if (manifest.format !== "surveyor-foam-v1") throw new Error(`unknown scene format ${manifest.format}`);
  onProgress?.(`fetching ${(manifest.bin_bytes / 1e6).toFixed(1)} MB`);
  const buf = await (await fetch(base + manifest.bin)).arrayBuffer();
  const S = manifest.sections;
  const { n, k, d } = manifest;
  const pos = view(buf, S.pos) as Float32Array;
  onProgress?.("uploading textures");
  const texPos = dataTexture(gl, pos, n, 4, gl.RGBA32F, gl.RGBA, gl.FLOAT, "pos");
  const texNSigma = dataTexture(gl, view(buf, S.nsigma), n, 4, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, "nsigma");
  const texSiteOff = dataTexture(gl, view(buf, S.siteoff), n * k, 4, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, "siteoff");
  const texSvAxis = dataTexture(gl, view(buf, S.svaxis), n * k * d, 3, gl.RGB16F, gl.RGB, gl.HALF_FLOAT, "svaxis");
  const texSvRgb = dataTexture(gl, view(buf, S.svrgb), n * k * d, 3, gl.RGB16F, gl.RGB, gl.HALF_FLOAT, "svrgb");
  const texAdjOff = dataTexture(gl, view(buf, S.adjoff), n + 1, 1, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, "adjoff");
  const texAdjIdx = dataTexture(gl, view(buf, S.adjidx), manifest.n_edges_directed, 1, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, "adjidx");
  return { manifest, n, k, d, pos: new Float32Array(pos), texPos, texNSigma, texSiteOff, texSvAxis, texSvRgb, texAdjOff, texAdjIdx };
}

/** benchmark.py:324 — start cell = argmin |p - eye|^2 - r^2 (brute force; ~0.1 ms per 100k cells). */
export function startCell(scene: FoamScene, eye: ArrayLike<number>): number {
  const p = scene.pos;
  let best = 0;
  let bestV = Infinity;
  const ex = eye[0], ey = eye[1], ez = eye[2];
  for (let i = 0, o = 0; i < scene.n; i++, o += 4) {
    const dx = p[o] - ex, dy = p[o + 1] - ey, dz = p[o + 2] - ez, r = p[o + 3];
    const v = dx * dx + dy * dy + dz * dz - r * r;
    if (v < bestV) { bestV = v; best = i; }
  }
  return best;
}
