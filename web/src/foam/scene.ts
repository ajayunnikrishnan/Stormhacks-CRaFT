/**
 * Loads an exported foam scene (tools/export_scene.py, format "craft-foam-v1")
 * and packs it into WebGL2 data textures. See docs/ARCHITECTURE.md §9 for the layout.
 */
import { dataTexture, type DataTexture } from "../render/gl";
import { parseScene, type SceneArrays, type SceneManifest, type RepoCamera } from "./sceneData";
export type { SceneManifest, RepoCamera } from "./sceneData";

export interface FoamScene extends SceneArrays {
  /** GPU textures. */
  texPos: DataTexture; // RGBA32F  N
  texNSigma: DataTexture; // RGBA16F  N
  texSiteOff: DataTexture; // RGBA16F  N*k   (offset xyz, height)
  texSvAxis: DataTexture; // RGB16F   N*k*D
  texSvRgb: DataTexture; // RGB16F   N*k*D
  texAdjOff: DataTexture; // R32UI    N+1
  texAdjIdx: DataTexture; // R32UI    E
  /** union adjacency over the curvature sweep (present when exported with --curved) */
  texAdjOffU?: DataTexture;
  texAdjIdxU?: DataTexture;
}

export async function loadScene(gl: WebGL2RenderingContext, url: string, onProgress?: (msg: string) => void): Promise<FoamScene> {
  onProgress?.("fetching manifest");
  const base = url.replace(/\/[^/]*$/, "/");
  const manifest = (await (await fetch(url)).json()) as SceneManifest;
  onProgress?.(`fetching ${(manifest.bin_bytes / 1e6).toFixed(1)} MB`);
  // Scenes over GitHub's 100 MB file limit are committed gzipped as scene.gz.bin (NOT .gz: servers
  // treat that extension as a pre-compressed asset and add Content-Encoding, which breaks fetch in
  // some browsers). Fall back to it when the raw file is missing and gunzip by magic bytes.
  let res = await fetch(base + manifest.bin);
  let buf: ArrayBuffer;
  if (res.ok && !(res.headers.get("content-type") ?? "").includes("text/html")) buf = await res.arrayBuffer();
  else {
    const gzUrl = base + manifest.bin.replace(/\.bin$/, ".gz.bin");
    onProgress?.(`fetching compressed scene`);
    res = await fetch(gzUrl);
    if (!res.ok) throw new Error(`scene data not found: ${base + manifest.bin} / ${gzUrl}`);
    buf = await res.arrayBuffer();
    const head = new Uint8Array(buf, 0, 2);
    if (head[0] === 0x1f && head[1] === 0x8b) {
      onProgress?.("decompressing");
      buf = await new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
    }
  }
  const a = parseScene(manifest, buf);
  const { n, k, d } = a;
  onProgress?.("uploading textures");
  const out: FoamScene = {
    ...a,
    texPos: dataTexture(gl, a.pos, n, 4, gl.RGBA32F, gl.RGBA, gl.FLOAT, "pos"),
    texNSigma: dataTexture(gl, a.nsigma, n, 4, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, "nsigma"),
    texSiteOff: dataTexture(gl, a.siteoff, n * k, 4, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, "siteoff"),
    texSvAxis: dataTexture(gl, a.svaxis, n * k * d, 3, gl.RGB16F, gl.RGB, gl.HALF_FLOAT, "svaxis"),
    texSvRgb: dataTexture(gl, a.svrgb, n * k * d, 3, gl.RGB16F, gl.RGB, gl.HALF_FLOAT, "svrgb"),
    texAdjOff: dataTexture(gl, a.adjOff, n + 1, 1, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, "adjoff"),
    texAdjIdx: dataTexture(gl, a.adjIdx, manifest.n_edges_directed, 1, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, "adjidx"),
  };
  if (a.adjOffU && a.adjIdxU && manifest.curved) {
    out.texAdjOffU = dataTexture(gl, a.adjOffU, n + 1, 1, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, "adjoff_u");
    out.texAdjIdxU = dataTexture(gl, a.adjIdxU, manifest.curved.n_edges_directed_union, 1, gl.R32UI, gl.RED_INTEGER, gl.UNSIGNED_INT, "adjidx_u");
  }
  return out;
}

/** benchmark.py:324 — start cell = argmin |p - eye|^2 - r^2 (brute force; ~0.1 ms per 100k cells). */
export function startCell(scene: SceneArrays, eye: ArrayLike<number>): number {
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
