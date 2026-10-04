#version 300 es
// Spherical Voronoi pre-pass, curved version (docs/WRITEUP.md §4, §3.11).
// For detail site S_ij = M_i · exp_o(s·off_ij): the geodesic from the camera o_w reaches S_ij
// with direction γ'(d) (Eq. 4) where γ'(0) = u = tangent toward S_ij (Eq. 5). The SV axes are
// stored in the cell's flat frame, so the lookup direction is (M_i⁻¹ γ'(d)).yzw normalised.
// For κ = 0 this is exactly normalize(S − eye) as in color_fn.py.
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

#include "common.glsl"
#include "geometry.glsl"

uniform sampler2D uA;        // RGBA32F N
uniform sampler2D uRad;      // RG32F   N : (R, cs R)
uniform sampler2D uSiteOff;  // RGBA16F N*k : flat offset (m), height
uniform sampler2D uSvAxis;   // RGB16F  N*k*D
uniform sampler2D uSvRgb;    // RGB16F  N*k*D
uniform int uK;
uniform int uD;
uniform int uCount;          // N*k
uniform int uKappa;
uniform float uScale;
uniform vec4 uRayO;          // camera position in world model coords

out vec4 oColor;

void main() {
  int idx = int(gl_FragCoord.y) * (TEX_W_MASK + 1) + int(gl_FragCoord.x);
  if (idx >= uCount) { oColor = vec4(0.5, 0.5, 0.5, 0.0); return; }
#ifdef KAPPA
  const int k = KAPPA;
#else
  int k = uKappa;
#endif
  int prim = idx / uK;
  vec4 ai = fetch4(uA, prim);
  vec2 rr = fetch4(uRad, prim).xy;
  vec4 P = (k == 0) ? vec4(1.0, ai.yzw) : (ai + vec4(1.0, 0.0, 0.0, 0.0)) * rr.y;   // texture stores a0 − 1
  mat4 M = translationToK(k, P);
  vec3 off = uScale * fetch4(uSiteOff, idx).xyz;
  vec4 S = M * embedPointK(k, off);                 // world position of the detail site
  float d;
  vec4 u = tangentTowardK(k, uRayO, S, d);          // Eq. (5)
  vec4 arrive = geodesicDirK(k, uRayO, u, d);       // Eq. (4): transported direction at S
  vec4 local = inverseIsometryK(k, M) * arrive;     // into the cell's flat frame
  vec3 dir = normalize(local.yzw);
  float wsum = 0.0;
  vec3 vsum = vec3(0.0);
  int base = idx * uD;
  for (int q = 0; q < 32; ++q) {
    if (q >= uD) break;
    vec3 raw = fetch4(uSvAxis, base + q).xyz;
    float temp = length(raw);
    vec3 axis = raw / max(temp, 1e-20);
    float w = exp(-temp * length(dir - axis));
    wsum += w;
    vsum += w * fetch4(uSvRgb, base + q).xyz;
  }
  oColor = vec4(max(vsum / wsum + 0.5, 0.0), 1.0);
}
