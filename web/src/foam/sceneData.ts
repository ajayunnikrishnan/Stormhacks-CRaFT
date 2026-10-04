/** GL-free parsing of an exported scene (shared by the WebGL loader and node tests). */

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
  curved?: { centre: number[]; k_max: number; k_neg?: number; scene_extent: number; n_edges_directed_union: number; sweep: Record<string, unknown> };
}

export interface SceneArrays {
  manifest: SceneManifest;
  n: number;
  k: number;
  d: number;
  pos: Float32Array; // [N,4] xyz, r
  nsigma: Uint16Array; // f16 bits [N,4]
  siteoff: Uint16Array; // f16 bits [N*k,4]
  svaxis: Uint16Array; // f16 bits [N*k*D,3]
  svrgb: Uint16Array;
  adjOff: Uint32Array;
  adjIdx: Uint32Array;
  adjOffU?: Uint32Array;
  adjIdxU?: Uint32Array;
}

export function view(buf: ArrayBuffer, s: Section): Float32Array | Uint16Array | Uint32Array {
  const count = s.shape.reduce((a, b) => a * b, 1);
  switch (s.dtype) {
    case "float32": return new Float32Array(buf, s.offset, count);
    case "float16": return new Uint16Array(buf, s.offset, count);
    case "uint32": return new Uint32Array(buf, s.offset, count);
    default: throw new Error(`unsupported dtype ${s.dtype}`);
  }
}

export function parseScene(manifest: SceneManifest, buf: ArrayBuffer): SceneArrays {
  if (manifest.format !== "craft-foam-v1") throw new Error(`unknown scene format ${manifest.format}`);
  const S = manifest.sections;
  const out: SceneArrays = {
    manifest, n: manifest.n, k: manifest.k, d: manifest.d,
    pos: view(buf, S.pos) as Float32Array,
    nsigma: view(buf, S.nsigma) as Uint16Array,
    siteoff: view(buf, S.siteoff) as Uint16Array,
    svaxis: view(buf, S.svaxis) as Uint16Array,
    svrgb: view(buf, S.svrgb) as Uint16Array,
    adjOff: view(buf, S.adjoff) as Uint32Array,
    adjIdx: view(buf, S.adjidx) as Uint32Array,
  };
  if (S.adjoff_u && S.adjidx_u) { out.adjOffU = view(buf, S.adjoff_u) as Uint32Array; out.adjIdxU = view(buf, S.adjidx_u) as Uint32Array; }
  return out;
}

/** Decode IEEE half bits to float (for tests / CPU use). */
export function halfToFloat(h: number): number {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}
