/** Loads GLSL sources at build time and resolves `#include "x.glsl"` against shaders/. */
import common from "../../shaders/common.glsl?raw";
import geometry from "../../shaders/geometry.glsl?raw";
import svPrepass from "../../shaders/sv_prepass.frag?raw";
import svPrepassCurved from "../../shaders/sv_prepass_curved.frag?raw";
import walkFlat from "../../shaders/walk_flat.frag?raw";
import walkCurved from "../../shaders/walk_curved.frag?raw";
import composite from "../../shaders/composite.frag?raw";

const includes: Record<string, string> = { "common.glsl": common, "geometry.glsl": geometry };

export function preprocess(src: string, defines: Record<string, string | number> = {}): string {
  let out = src.replace(/^#include\s+"([^"]+)"\s*$/gm, (_, name: string) => {
    const inc = includes[name];
    if (inc === undefined) throw new Error(`unknown include ${name}`);
    return `// ---- begin ${name} ----\n${inc}\n// ---- end ${name} ----`;
  });
  const defs = Object.entries(defines).map(([k, v]) => `#define ${k} ${v}`).join("\n");
  if (defs) out = out.replace(/^(#version[^\n]*\n)/, `$1${defs}\n`);
  return out;
}

/** Per-κ compiled variants of the curved shaders (κ folded to a constant). */
export function curvedVariant(kappa: -1 | 0 | 1) {
  const d = { KAPPA: kappa === -1 ? "(-1)" : String(kappa) };
  return { svPrepassCurved: preprocess(svPrepassCurved, d), walkCurved: preprocess(walkCurved, d) };
}

export const SHADERS = {
  svPrepass: preprocess(svPrepass),
  svPrepassCurved: preprocess(svPrepassCurved),
  walkFlat: preprocess(walkFlat),
  walkCurved: preprocess(walkCurved),
  composite: preprocess(composite),
};
