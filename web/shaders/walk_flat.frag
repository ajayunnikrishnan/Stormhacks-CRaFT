#version 300 es
// Flat-space foam ray walk. Faithful port of powerfoam/raytrace.py:benchmark_kernel
// (+ rendering_math.py, plane_intersection_fwd_local). See docs/ARCHITECTURE.md §3, §5
// for the derivation of every line. Phase 1: Euclidean space only; the curved
// walker (Eqs. 3–7) replaces the ray/plane/sphere primitives in Phase 3.
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

#include "common.glsl"

#define MAX_STEPS 512       // cells visited per ray (paper: ~37 average with Steiner points)
#define MAX_NEIGHBOURS 1024 // per-cell adjacency cap (loop breaks at the real degree)
#define MAX_K 8             // detail sites per cell (repo max_sites = 8)
#define SOFT_VORONOI_TEMP 10.0  // raytrace.py:39  temp = wp.constant(10.0)

uniform sampler2D  uPos;      // RGBA32F N     : p.xyz, r
uniform sampler2D  uNSigma;   // RGBA16F N     : n.xyz, sigma
uniform sampler2D  uSiteOff;  // RGBA16F N*k   : offset.xyz, height
uniform sampler2D  uSiteRgb;  // RGBA16F N*k   : SV colour from the pre-pass
uniform usampler2D uAdjOff;   // R32UI   N+1
uniform usampler2D uAdjIdx;   // R32UI   E
uniform int   uK;
uniform int   uStart;         // start cell (argmin power distance from the eye)
uniform float uThreshold;     // transmittance cutoff (benchmark.py: 1e-2)
uniform bool  uNearCull;      // raytrace.py:174  |p - eye| < 4r  =>  hit = false
uniform bool  uRepoPixelGrid; // true: x = 2j/(W-1) - 1 (camera.py); false: pixel centres
uniform vec2  uResolution;
uniform vec3  uEye;
uniform vec3  uRight;         // scaled by tan(fov_x/2) like TorchCamera.right
uniform vec3  uUp;            // scaled by tan(fov_y/2)
uniform vec3  uForward;       // unit, = normalize(cross(up, right))
uniform vec3  uBackground;

out vec4 oColor;

// rendering_math.py:5 ray_sphere_intersect — returns (hit, t_near, t_far)
bool raySphere(vec3 eye, vec3 d, vec3 c, float r, out float tn, out float tf) {
  vec3 oc = eye - c;
  float qb = 2.0 * dot(oc, d);
  float qc = dot(oc, oc) - r * r;
  float disc = qb * qb - 4.0 * qc;
  tn = 0.0; tf = 0.0;
  if (disc < 0.0) return false;
  float s = sqrt(disc);
  tf = (-qb + s) * 0.5;
  tn = (-qb - s) * 0.5;
  if (tn < 0.0 && tf < 0.0) return false;
  if (tn < 0.0) { tn = 0.0; }
  return true;
}

// rendering_math.py:129 ray_plane_intersect — plane n·x = n·p + h; returns t, dp = n·d
float rayPlane(vec3 eye, vec3 d, vec3 p, vec3 n, float h, out float dp) {
  dp = dot(n, d);
  return (dot(p - eye, n) + h) / dp;
}

void main() {
  // camera.py:get_ray_dir. Fragment rows run bottom-up; the repo's i runs top-down.
  float j = floor(gl_FragCoord.x);
  float i = uResolution.y - 1.0 - floor(gl_FragCoord.y);
  float x, y;
  if (uRepoPixelGrid) {
    x = 2.0 * j / (uResolution.x - 1.0) - 1.0;
    y = 1.0 - 2.0 * i / (uResolution.y - 1.0);
  } else {
    x = 2.0 * (j + 0.5) / uResolution.x - 1.0;
    y = 1.0 - 2.0 * (i + 0.5) / uResolution.y;
  }
  vec3 d = normalize(x * uRight + y * uUp + uForward);
  vec3 eye = uEye;

  vec3 rgb = vec3(0.0);
  float logT = 0.0;
  int prim = uStart;
  float ptNear = 0.0;

  for (int step = 0; step < MAX_STEPS; ++step) {
    float trans = exp(logT);
    if (trans < uThreshold || prim == INT_MAX_ID) break;

    vec4 pr = fetch4(uPos, prim);
    vec3 c = pr.xyz;
    float r = pr.w;

    float tNear, tFar;
    bool hit = raySphere(eye, d, c, r, tNear, tFar);
    if (uNearCull && length(c - eye) < 4.0 * r) hit = false;

    int a = int(fetchU(uAdjOff, prim));
    int b = int(fetchU(uAdjOff, prim + 1));
    int next = INT_MAX_ID;
    float ptFar = 1e10;
    vec3 oc = eye - c;
    for (int q = 0; q < MAX_NEIGHBOURS; ++q) {
      if (a + q >= b) break;
      int jn = int(fetchU(uAdjIdx, a + q));
      vec4 pj = fetch4(uPos, jn);
      // Radical plane of spheres (c,r),(pj,rj):  diff·x = ½(|pj|²−|c|² + r²−rj²)   (rendering_math.py:63)
      // Evaluated relative to c for fp32 conditioning (same plane, same math):
      //   diff·(x − c) = ½(|diff|² + r² − rj²)
      vec3 diff = pj.xyz - c;
      float rhs = 0.5 * (dot(diff, diff) + r * r - pj.w * pj.w);
      float dp = dot(d, diff);
      float tFace = (rhs - dot(oc, diff)) / dp;
      if (dp >= 0.0) {
        if (tFace < ptFar) { next = jn; ptFar = tFace; }  // first exit face (raytrace.py:195)
        tFar = min(tFace, tFar);
      } else {
        tNear = max(tFace, tNear);
      }
    }

    vec4 ns = fetch4(uNSigma, prim);
    vec3 n = ns.xyz;
    float sigma = ns.w;
    if (!hit || tNear > tFar || sigma < 1e-3) {      // raytrace.py:208
      prim = next;
      ptNear = max(ptNear, ptFar);
      continue;
    }

    // ---- plane_intersection_fwd_local (raytrace.py:43) ----
    float dp0;
    float tSurf0 = rayPlane(eye, d, c, n, 0.0, dp0);
    float tq0 = (dp0 >= 0.0) ? tNear : max(tNear, tSurf0);
    vec3 xq0 = eye + tq0 * d;
    float invR2 = 1.0 / (r * r);
    int sbase = prim * uK;
    float hsum = 0.0, wsum0 = 0.0;
    for (int s = 0; s < MAX_K; ++s) {
      if (s >= uK) break;
      vec4 so = fetch4(uSiteOff, sbase + s);
      vec3 site = c + so.xyz;
      vec3 dv = xq0 - site;
      float w = exp(-SOFT_VORONOI_TEMP * dot(dv, dv) * invR2);
      hsum += w * so.w;
      wsum0 += w;
    }
    float height = hsum / max(wsum0, 1e-20);

    float dp1;
    float tSurf = rayPlane(eye, d, c, n, height, dp1);
    float tq1 = (dp1 >= 0.0) ? tNear : max(tNear, tSurf);
    vec3 xq1 = eye + tq1 * d;
    vec3 csum = vec3(0.0);
    float wsum1 = 0.0;
    for (int s = 0; s < MAX_K; ++s) {
      if (s >= uK) break;
      vec4 so = fetch4(uSiteOff, sbase + s);
      vec3 site = c + so.xyz;
      vec3 dv = xq1 - site;
      float w = exp(-SOFT_VORONOI_TEMP * dot(dv, dv) * invR2);
      csum += w * fetch4(uSiteRgb, sbase + s).rgb;
      wsum1 += w;
    }
    vec3 colour = csum / max(wsum1, 1e-20);
    // ---- end plane_intersection_fwd_local ----

    if (dp1 >= 0.0) tFar = min(tSurf, tFar); else tNear = max(tSurf, tNear);  // raytrace.py:225

    prim = next;
    ptNear = max(ptNear, ptFar);
    float dt = tFar - tNear;
    if (hit && dt > 0.0) {
      float delta = -sigma * dt;              // rendering_math.py:150 density_integral
      float alpha = 1.0 - exp(delta);
      rgb += colour * alpha * trans;
      logT += delta;
    }
  }
  rgb += uBackground * exp(logT);
  oColor = vec4(rgb, 1.0);
}
