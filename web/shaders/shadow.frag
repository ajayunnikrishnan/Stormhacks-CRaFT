#version 300 es
// Shadow rays (§3.9): from the primary hit point X toward each light, a geodesic walk through
// the same foam accumulating transmittance. Traversal identical to walk_curved.frag (Eqs. 6–7,
// union adjacency); the dense part of a cell is its ball ∩ the half-space behind the undisplaced
// dipole plane ⟨x, N_i⟩ ≤ 0 (exact, Eq. 6) — the detail-site displacement is ignored here.
// Lights inside a convex domain are reached without crossing domain faces (geodesic convexity),
// so no pairings are applied. Output: transmittance toward lights 0..3 in RGBA.
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

#include "common.glsl"
#include "geometry.glsl"

#define MAX_STEPS 512
#define MAX_NEIGHBOURS 1024

uniform sampler2D  uA;
uniform sampler2D  uRad;
uniform sampler2D  uNSigma;
uniform usampler2D uAdjOff;
uniform usampler2D uAdjIdx;
uniform sampler2D  uGHit;     // RGBA32F: hit position (world model), w = 1 if hit
uniform sampler2D  uGMeta;    // RGBA32F: (t_hit, cell id, hops, flags)
uniform int   uKappa;
uniform float uScale;
uniform int   uLightCount;
uniform vec4  uLightPos[4];
uniform vec2  uGRes;          // G-buffer resolution (this pass may run at lower res)
uniform vec2  uRes;

out vec4 oShadow;

float transmittanceToward(int k, float kk, vec4 o, vec4 L, int startCell) {
  float dL;
  vec4 v = tangentTowardK(k, o, L, dL);
  if (dL < 1e-5) return 1.0;
  float tEnd = dL - 2e-3;
  float logT = 0.0;
  int prim = startCell;
  float ptNear = 0.0;
  for (int step = 0; step < MAX_STEPS; ++step) {
    if (prim == INT_MAX_ID || ptNear >= tEnd || logT < -7.0) break;
    vec4 ai = fetch4(uA, prim);
    vec2 rr = fetch4(uRad, prim).xy;
    float R = rr.x;
    vec4 P = (k == 0) ? vec4(1.0, ai.yzw) : (ai + vec4(1.0, 0.0, 0.0, 0.0)) * rr.y;
    float tb1, tb2;
    bool hit = ballIntervalK(k, o, v, P, R, ptNear, tb1, tb2);
    if (hit && tb2 < 0.0) hit = false;
    float tNear = max(tb1, 0.0), tFar = tb2;
    int a = int(fetchU(uAdjOff, prim)), b = int(fetchU(uAdjOff, prim + 1));
    int next = INT_MAX_ID;
    float ptFar = 1e30;
    for (int q = 0; q < MAX_NEIGHBOURS; ++q) {
      if (a + q >= b) break;
      int jn = int(fetchU(uAdjIdx, a + q));
      vec4 da = fetch4(uA, jn) - ai;
      vec4 w = vec4(kk * da.x, da.yzw);
      float A = dot(w, o), B = dot(w, v);
      float te, tn;
      planeExitEntryK(k, A, B, ptNear, te, tn);
      if (te < ptFar) { next = jn; ptFar = te; }
      tFar = min(te, tFar);
      tNear = max(tn, tNear);
    }
    if (ptFar >= 1e29) next = INT_MAX_ID;
    vec4 ns = fetch4(uNSigma, prim);
    float sigma = ns.w / uScale;
    tFar = min(tFar, tEnd);
    if (hit && tNear < tFar && ns.w >= 1e-3) {
      // dense half-space behind the dipole plane through P with transported normal N = T_P(0, n̄)
      mat4 M = translationToK(k, P);
      vec4 N = M * vec4(0.0, ns.xyz);
      vec4 wn = (k == 0) ? vec4(-dot(N.yzw, P.yzw), N.yzw) : vec4(float(k) * N.x, N.yzw);
      float A = dot(wn, o), B = dot(wn, v);
      float te, tn;
      planeExitEntryK(k, A, B, tNear, te, tn);
      // on the dense side at tNear?  value ≤ 0 ⇒ inside; else the dense part starts at the next entry
      float val = planeValueK(k, A, B, tNear);
      float dNear = tNear, dFar = tFar;
      if (val <= 0.0) dFar = min(dFar, te); else dNear = max(dNear, tn > tNear ? tn : planeExitAfterK(k, -A, -B, tNear));
      float dt = dFar - dNear;
      if (dt > 0.0) logT -= sigma * dt;
    }
    prim = next;
    ptNear = max(ptNear, ptFar);
  }
  return exp(logT);
}

void main() {
  vec2 uv = gl_FragCoord.xy / uRes;
  ivec2 gij = ivec2(uv * uGRes);
  vec4 hit = texelFetch(uGHit, gij, 0);
  vec4 meta = texelFetch(uGMeta, gij, 0);
  if (hit.w < 0.5) { oShadow = vec4(1.0); return; }
  int k = uKappa;
  float kk = (k == 0) ? 1.0 : float(k);
  int cell = int(meta.y + 0.5);
  vec4 o = projectK(k, vec4(meta.w, hit.xyz)); // G-buffer stores (x1,x2,x3) in hit.xyz and x0 in meta.w
  vec4 s = vec4(1.0);
  for (int i = 0; i < 4; ++i) {
    if (i >= uLightCount) break;
    s[i] = transmittanceToward(k, kk, o, uLightPos[i], cell);
  }
  oShadow = s;
}
