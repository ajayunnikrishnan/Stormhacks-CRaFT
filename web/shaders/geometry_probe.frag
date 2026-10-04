#version 300 es
// Test probe: evaluates shaders/geometry.glsl on per-texel inputs so the harness can
// compare the GLSL (fp32) implementation with src/geometry/space.ts (fp64).
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

#include "common.glsl"
#include "geometry.glsl"

uniform sampler2D uO;      // RGBA32F: ray origin o
uniform sampler2D uV;      // RGBA32F: ray direction v
uniform sampler2D uW;      // RGBA32F: plane covector w
uniform sampler2D uP;      // RGBA32F: ball centre / target point p
uniform sampler2D uParam;  // RGBA32F: (t, r, tMin, kappa)
uniform int uMode;
uniform int uCount;

out vec4 oColor;

void main() {
  int idx = int(gl_FragCoord.y) * (TEX_W_MASK + 1) + int(gl_FragCoord.x);
  if (idx >= uCount) { oColor = vec4(0.0); return; }
  vec4 o = fetch4(uO, idx), v = fetch4(uV, idx), w = fetch4(uW, idx), p = fetch4(uP, idx), prm = fetch4(uParam, idx);
  float t = prm.x, r = prm.y, tMin = prm.z;
  int k = int(prm.w);
  float A = dot(w, o), B = dot(w, v);
  if (uMode == 0) { oColor = geodesicK(k, o, v, t); return; }
  if (uMode == 1) { oColor = geodesicDirK(k, o, v, t); return; }
  if (uMode == 2) { float d; oColor = tangentTowardK(k, o, p, d); return; }
  if (uMode == 3) {
    float d = distK(k, o, p);
    float tr; bool ex; bool has = planeRootK(k, A, B, tr, ex);
    oColor = vec4(d, has ? tr : 0.0, ex ? 1.0 : 0.0, has ? 1.0 : 0.0); return;
  }
  if (uMode == 4) {
    float t1, t2; bool has = ballIntervalK(k, o, v, p, r, tMin, t1, t2);
    float ea = planeExitAfterK(k, A, B, tMin);
    oColor = vec4(has ? 1.0 : 0.0, has ? t1 : 0.0, has ? t2 : 0.0, ea > 1e29 ? 1e30 : ea); return;
  }
  if (uMode == 5) {
    vec4 e = embedPointK(k, p.yzw);
    oColor = e; return;
  }
  if (uMode == 6) {
    mat4 T = translationToK(k, p);
    oColor = T * v; return;
  }
  if (uMode == 7) {
    mat4 T = translationToK(k, p);
    mat4 Ti = inverseIsometryK(k, T);
    oColor = Ti * (T * v); return; // should equal v
  }
  if (uMode == 8) {
    float eb = planeEntryBeforeK(k, A, B, tMin + 10.0);
    vec3 l = logAtOriginK(k, p);
    oColor = vec4(eb < -1e29 ? -1e30 : eb, l); return;
  }
  oColor = vec4(0.0);
}
