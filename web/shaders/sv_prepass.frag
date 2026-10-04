#version 300 es
// Spherical Voronoi pre-pass: one fragment per detail site (N*k texels).
// Port of powerfoam/color_fn.py:spherical_voronoi_fwd_kernel (docs/ARCHITECTURE.md §4):
//   dir = normalize(site - eye)
//   w_d = exp(-temp_d * |dir - axis_d|),  temp_d = |raw_axis_d|, axis_d = raw_axis_d / temp_d
//   rgb = max(sum(w_d rgb_d) / sum(w_d) + 0.5, 0)
// The repo's frustum cull (grey 0.5 outside the camera FOV) is intentionally NOT applied:
// in the game, cells are seen from all directions.
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

#include "common.glsl"

uniform sampler2D uPos;      // RGBA32F  N      : p.xyz, r
uniform sampler2D uSiteOff;  // RGBA16F  N*k    : offset.xyz, height
uniform sampler2D uSvAxis;   // RGB16F   N*k*D  : raw axis
uniform sampler2D uSvRgb;    // RGB16F   N*k*D
uniform int uK;
uniform int uD;
uniform int uCount;          // N*k
uniform vec3 uEye;

out vec4 oColor;

void main() {
  int idx = int(gl_FragCoord.y) * (TEX_W_MASK + 1) + int(gl_FragCoord.x);
  if (idx >= uCount) { oColor = vec4(0.5, 0.5, 0.5, 0.0); return; }
  int prim = idx / uK;
  vec3 site = fetch4(uPos, prim).xyz + fetch4(uSiteOff, idx).xyz;
  vec3 dir = normalize(site - uEye);
  float wsum = 0.0;
  vec3 vsum = vec3(0.0);
  int base = idx * uD;
  for (int d = 0; d < 32; ++d) {
    if (d >= uD) break;
    vec3 raw = fetch4(uSvAxis, base + d).xyz;
    float temp = length(raw);
    vec3 axis = raw / max(temp, 1e-20);
    float w = exp(-temp * length(dir - axis));
    wsum += w;
    vsum += w * fetch4(uSvRgb, base + d).xyz;
  }
  vec3 rgb = max(vsum / wsum + 0.5, 0.0);
  oColor = vec4(rgb, 1.0);
}
