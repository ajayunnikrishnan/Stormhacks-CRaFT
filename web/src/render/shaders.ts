/** Loads GLSL sources at build time and resolves `#include "x.glsl"` against shaders/. */
import common from "../../shaders/common.glsl?raw";
import svPrepass from "../../shaders/sv_prepass.frag?raw";
import walkFlat from "../../shaders/walk_flat.frag?raw";
import composite from "../../shaders/composite.frag?raw";

const includes: Record<string, string> = { "common.glsl": common };

export function preprocess(src: string): string {
  return src.replace(/^#include\s+"([^"]+)"\s*$/gm, (_, name: string) => {
    const inc = includes[name];
    if (inc === undefined) throw new Error(`unknown include ${name}`);
    return `// ---- begin ${name} ----\n${inc}\n// ---- end ${name} ----`;
  });
}

export const SHADERS = {
  svPrepass: preprocess(svPrepass),
  walkFlat: preprocess(walkFlat),
  composite: preprocess(composite),
};
