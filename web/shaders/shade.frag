#version 300 es
// Deferred shading + composite (§3.9): C = R_baked ⊙ (ambient + Σ_i E_i·ρ·shadow_i) with Eq. (8)
// irradiance (S³: + the antipodal term), the flashlight as a spotlight at the camera (its shadow
// is the primary ray's own transmittance up to the hit), visible lamp glows, fog exp(−σt), and a
// Reinhard tone map. Lighting is evaluated once per pixel at the median-depth hit (documented
// approximation: semi-transparent edges inherit the surface's lighting).
precision highp float;
precision highp int;
precision highp sampler2D;

#include "common.glsl"
#include "geometry.glsl"

uniform sampler2D uGColor;   // baked rgb, w = final transmittance
uniform sampler2D uGHit;     // (x1,x2,x3, hitFlag)
uniform sampler2D uGNormal;  // transported normal (x0..x3)
uniform sampler2D uGMeta;    // (t_hit, cell, T_before_hit, x0)
uniform sampler2D uShadow;   // transmittance toward lights 0..3 (may be lower res; sampled linearly)
uniform vec2  uRes;          // output resolution
uniform vec2  uGRes;         // G-buffer resolution
uniform int   uKappa;
uniform float uScale;
uniform int   uLightCount;
uniform vec4  uLightPos[8];
uniform vec3  uLightColor[8]; // colour · power
uniform float uLightRadius;  // visible lamp radius (model units)
uniform int   uFlashOn;
uniform vec4  uCamPos;
uniform vec4  uCamFwd;       // unit tangent at the camera, world
uniform vec3  uFlashColor;   // colour · power
uniform vec2  uFlashCone;    // (cos inner, cos outer)
uniform float uAmbient;
uniform float uRho;
uniform float uFogSigma;     // 1/unit
uniform vec3  uFogColor;
uniform float uExposure;
uniform int   uLightingOn;   // 0 = baked only (reference mode)
uniform vec2  uTanHalfFov;
uniform mat4  uInvW;

out vec4 oColor;

// closest approach of the pixel's geodesic (o,v) to point L: returns (d_min, t_at_min)
vec2 closestApproach(int k, vec4 o, vec4 v, vec4 L) {
  if (k == 0) {
    vec3 w = L.yzw - o.yzw;
    float t = dot(w, v.yzw);
    return vec2(length(w - t * v.yzw), t);
  }
  float A = formK(k, o, L), B = formK(k, v, L);
  if (k < 0) {
    // cosh d(t) = −(A cosh t + B sinh t); min at tanh t = −B/A; min value √(A²−B²)
    float t = (abs(B) < abs(A)) ? atanh(-B / A) : 0.0;
    float c = sqrt(max(1.0, A * A - B * B));
    return vec2(acosh(c), t);
  }
  float R = length(vec2(A, B));
  float t = atan(B, A);
  t -= TWO_PI * floor(t / TWO_PI);
  return vec2(acos(clamp(R, -1.0, 1.0)), t);
}

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  ivec2 gij = ivec2(uv * uGRes);
  vec4 gc = texelFetch(uGColor, gij, 0);
  vec4 gh = texelFetch(uGHit, gij, 0);
  vec4 gn = texelFetch(uGNormal, gij, 0);
  vec4 gm = texelFetch(uGMeta, gij, 0);
  int k = uKappa;
  vec3 baked = gc.rgb;
  float Tfinal = gc.a;
  bool hasHit = gh.w > 0.5;
  vec3 col = baked;
  float tHit = hasHit ? gm.x : 1e30;

  if (uLightingOn == 1 && hasHit) {
    vec4 X = projectK(k, vec4(gm.w, gh.xyz));
    vec4 N = gn;
    vec4 sh = texture(uShadow, uv);
    // the baked colour already contains the capture-time lighting; it plays the role of albedo here
    vec3 lit = baked * uAmbient;
    for (int i = 0; i < 8; ++i) {
      if (i >= uLightCount) break;
      vec4 E = irradianceK(k, X, N, uLightPos[i], 1.0);
      float s = (i < 4) ? sh[i] : 1.0;
      // fog absorbs along the light path too: e^{−σd} direct, e^{−σ(2π−d)} the long way round (S³)
      float att = exp(-uFogSigma * E.z);
      float attAnti = exp(-uFogSigma * (TWO_PI - E.z));
      lit += baked * uRho * uLightColor[i] * (E.x * s * att + E.y * attAnti);   // antipodal term unshadowed
    }
    // flashlight: spotlight at the camera along the view direction; its shadow is the primary
    // ray's own transmittance up to the hit (gm.z)
    if (uFlashOn == 1) {
      vec4 E = irradianceK(k, X, N, uCamPos, 1.0);
      float dcam;
      vec4 u0 = tangentTowardK(k, uCamPos, X, dcam);     // leaving the camera toward X
      float cone = smoothstep(uFlashCone.y, uFlashCone.x, formK(k, u0, uCamFwd));
      lit += baked * uRho * uFlashColor * E.x * cone * gm.z * exp(-uFogSigma * E.z);
    }
    col = lit;
  }

  // visible lamps: glow where the pixel's geodesic passes within uLightRadius of a light before the hit
  if (uLightingOn == 1) {
    float px = 2.0 * gl_FragCoord.x / uRes.x - 1.0, py = 2.0 * gl_FragCoord.y / uRes.y - 1.0;
    vec3 dcam = normalize(vec3(px * uTanHalfFov.x, py * uTanHalfFov.y, -1.0));
    vec4 v = uInvW * vec4(0.0, dcam);
    for (int i = 0; i < 8; ++i) {
      if (i >= uLightCount) break;
      vec2 ca = closestApproach(k, uCamPos, v, uLightPos[i]);
      if (ca.y > 0.0 && ca.y < tHit) {
        float g = exp(-(ca.x * ca.x) / (uLightRadius * uLightRadius));
        col += uLightColor[i] * g * 2.0 + vec3(g * g);
      }
    }
  }

  // fog (also the walk's distance cutoff)
  float fogT = exp(-uFogSigma * min(tHit, 1e6));
  col = mix(uFogColor, col, fogT);
  // Reinhard tone map
  col *= uExposure;
  col = col / (1.0 + col);
  oColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
