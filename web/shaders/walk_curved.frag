#version 300 es
// Curved-space foam ray walk (κ ∈ {−1,0,+1}) — docs/WRITEUP.md §3.4–3.6, §3.11.
// Same control flow as walk_flat.frag (= powerfoam/raytrace.py:benchmark_kernel), with the
// Euclidean primitives replaced by the closed-form geodesic ones:
//   • traversal is EXACT: ball interval Eq. (7), bisector crossings Eq. (6) with the curved
//     power-cell rule a_i = p_i / cs_κ(r_i)  (cell = argmax ⟨x,a_i⟩', faces ⟨x,a_j−a_i⟩' = 0);
//   • shading is LOCALLY FLAT: the ray segment is mapped into the cell's frame (M_p⁻¹), log-mapped
//     to tangent coordinates, and the unmodified dipole / detail-site code runs there. The
//     undisplaced dipole plane is exact in these coordinates (a totally geodesic plane through
//     the frame origin is linear in the log map); the only approximation is treating the ray
//     as straight within the cell, error O(|κ|·R²), measured in tests/.
// For κ = 0 (E³) every formula reduces to walk_flat.frag; the harness checks the two agree.
precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

#include "common.glsl"
#include "geometry.glsl"

#define MAX_STEPS 2048
#define MAX_NEIGHBOURS 1024
#define MAX_K 8
#define SOFT_VORONOI_TEMP 10.0   // raytrace.py:39

uniform sampler2D  uA;        // RGBA32F N   : a_i (E³: (−pm_i, p̄_i)), unit-model coords, centred
uniform sampler2D  uRad;      // RG32F   N   : (R_i, cs_κ(R_i)), R_i = s·r_i
uniform sampler2D  uNSigma;   // RGBA16F N   : flat normal n̄_i (scene frame), σ_i (1/m)
uniform sampler2D  uSiteOff;  // RGBA16F N*k : flat offset (m), height (m)
uniform sampler2D  uSiteRgb;  // RGBA16F N*k : SV colour from sv_prepass_curved
uniform usampler2D uAdjOff;   // R32UI   N+1 (union adjacency over the κ sweep)
uniform usampler2D uAdjIdx;   // R32UI   E
uniform int   uKappa;         // κ
uniform float uScale;         // s = √|k| (1 for E³): metres → unit model
uniform int   uK;
uniform int   uStart;
uniform float uThreshold;
uniform bool  uNearCull;      // raytrace.py:174 parity: d(camera, p_i) < 4 R_i ⇒ no shading
uniform float uTMax;          // stop after this arc length (S³: 2π)
uniform vec2  uResolution;
uniform vec2  uTanHalfFov;    // (tan fovx/2, tan fovy/2)
uniform bool  uRepoPixelGrid; // camera.py grid (edges at ±1) for parity tests; else pixel centres
uniform vec4  uRayO;          // camera position in world model coords = W⁻¹ o
uniform mat4  uInvW;          // camera → world isometry (W⁻¹)
uniform vec3  uBackground;

// ---- topology (§3.8): fundamental domain faces and pairings ----
#define MAX_FACES 12
#define LOC_GRID 32
uniform int   uFaceCount;              // 0 = open space
uniform vec4  uFaceW[MAX_FACES];       // covectors: inside ⇔ w·x ≤ 0
uniform mat4  uFaceG[MAX_FACES];       // isometry applied to a ray exiting face f (maps f → partner)
uniform mat4  uFaceLocInv[MAX_FACES];  // world → PARTNER face chart frame (for the lookup grid)
uniform int   uFacePartner[MAX_FACES];
uniform float uFaceChartHalf[MAX_FACES]; // chart half-extent of the PARTNER face
uniform int   uMaxHops;
uniform usampler2D uLocGrid;           // R32UI: face*G*G + j*G + i → cell id

float cellValue(float kk, vec4 a, vec4 x) { return kk * a.x * x.x + dot(a.yzw, x.yzw); }

// Steepest ascent on ⟨x,a_i⟩' from a guess (grid entry); a handful of steps in practice.
int locateCell(int guess, vec4 x, float kk) {
  int c = guess;
  float best = cellValue(kk, fetch4(uA, c), x);
  for (int it = 0; it < 32; ++it) {
    int a = int(fetchU(uAdjOff, c)), b = int(fetchU(uAdjOff, c + 1));
    int bj = -1;
    for (int q = 0; q < MAX_NEIGHBOURS; ++q) {
      if (a + q >= b) break;
      int j = int(fetchU(uAdjIdx, a + q));
      float vj = cellValue(kk, fetch4(uA, j), x);
      if (vj > best + 1e-7) { best = vj; bj = j; }
    }
    if (bj < 0) break;
    c = bj;
  }
  return c;
}

layout(location = 0) out vec4 oColor;   // baked rgb, w = final transmittance
layout(location = 1) out vec4 oHit;     // (x1, x2, x3, hitFlag) of the median-depth hit (world model coords)
layout(location = 2) out vec4 oNormal;  // transported dipole normal at the hit (ambient tangent)
layout(location = 3) out vec4 oMeta;    // (t_hit total arc length, cell id, transmittance before the hit, x0)

void main() {
  // pixel → camera-space direction (camera looks down −z, pixel centres)
  float px, py;
  if (uRepoPixelGrid) {
    px = 2.0 * floor(gl_FragCoord.x) / (uResolution.x - 1.0) - 1.0;
    py = 2.0 * floor(gl_FragCoord.y) / (uResolution.y - 1.0) - 1.0;
  } else {
    px = 2.0 * gl_FragCoord.x / uResolution.x - 1.0;
    py = 2.0 * gl_FragCoord.y / uResolution.y - 1.0;
  }
  vec3 dcam = normalize(vec3(px * uTanHalfFov.x, py * uTanHalfFov.y, -1.0));
#ifdef KAPPA
  const int k = KAPPA;          // compile-time variant: every κ branch folds away
#else
  int k = uKappa;
#endif
  float kk = (k == 0) ? 1.0 : float(k);           // κ' for the cell-rule form
  vec4 o = uRayO;
  vec4 v = uInvW * vec4(0.0, dcam);                // unit tangent at o (isometry image of a unit tangent at the origin)

  vec3 rgb = vec3(0.0);
  float logT = 0.0;
  bool gotHit = false;
  oHit = vec4(0.0); oNormal = vec4(0.0); oMeta = vec4(1e30, -1.0, 1.0, 1.0);
  int prim = uStart;
  float ptNear = 0.0;                              // arc length (since the last crossing) at which the ray entered `prim`
  float tTotal = 0.0;                              // arc length accumulated over previous domain hops
  int hops = 0;

  for (int step = 0; step < MAX_STEPS; ++step) {
    float trans = exp(logT);
    if (trans < uThreshold || prim == INT_MAX_ID || tTotal + ptNear > uTMax) break;

    vec4 ai = fetch4(uA, prim);
    vec2 rr = fetch4(uRad, prim).xy;
    float R = rr.x;
    vec4 P = (k == 0) ? vec4(1.0, ai.yzw) : (ai + vec4(1.0, 0.0, 0.0, 0.0)) * rr.y;   // texture stores a0 − 1   // p_i = a_i · cs(R_i)

    // Eq. (7): ball interval along the ray (first interval ending at or after ptNear)
    float tb1, tb2;
    bool hit = ballIntervalK(k, o, v, P, R, ptNear, tb1, tb2);
    if (hit && tb2 < 0.0) hit = false;             // rendering_math.py:18 (both roots behind)
    float tNear = max(tb1, 0.0), tFar = tb2;        // rendering_math.py:21 (origin inside ⇒ 0)
    if (uNearCull && distK(k, o, P) < 4.0 * R) hit = false;

    // Eq. (6): bisector planes with every neighbour: w = J'(a_j − a_i), cell side w·x ≤ 0
    int a = int(fetchU(uAdjOff, prim));
    int b = int(fetchU(uAdjOff, prim + 1));
    int next = INT_MAX_ID;
    float ptFar = 1e30;
    for (int q = 0; q < MAX_NEIGHBOURS; ++q) {
      if (a + q >= b) break;
      int jn = int(fetchU(uAdjIdx, a + q));
      vec4 da = fetch4(uA, jn) - ai;
      vec4 w = vec4(kk * da.x, da.yzw);
      float A = dot(w, o), B = dot(w, v);
      float te, tn;
      planeExitEntryK(k, A, B, ptNear, te, tn);      // first exit after / last entry before entering the cell
      if (te < ptFar) { next = jn; ptFar = te; }
      tFar = min(te, tFar);
      tNear = max(tn, tNear);
    }
    if (ptFar >= 1e29) next = INT_MAX_ID;

    // domain faces: does the ray leave the fundamental domain before leaving the cell?
    int exitFace = -1;
    float tDom = 1e30;
    for (int f = 0; f < MAX_FACES; ++f) {
      if (f >= uFaceCount) break;
      float te = planeExitAfterK(k, dot(uFaceW[f], o), dot(uFaceW[f], v), ptNear);
      if (te < tDom) { tDom = te; exitFace = f; }
    }
    bool crossing = exitFace >= 0 && tDom < ptFar;
    if (crossing) { tFar = min(tFar, tDom); ptFar = tDom; }

    vec4 ns = fetch4(uNSigma, prim);
    vec3 nflat = ns.xyz;
    float sigma = ns.w / uScale;                    // 1/m → 1/unit
    bool shade = hit && tNear <= tFar && sigma * uScale >= 1e-3;   // raytrace.py:208 (σ test in 1/m like the repo)
    if (shade) {

    // ---- locally flat shading in the cell frame (§3.11) ----
    mat4 Minv = inverseIsometryK(k, translationToK(k, P));
    vec4 xin = Minv * geodesicK(k, o, v, tNear);     // entry point, cell frame
    vec4 din = Minv * geodesicDirK(k, o, v, tNear);  // transported direction at the entry
    vec3 e3 = logAtOriginK(k, xin);                   // tangent coordinates of the entry point
    vec3 d3 = normalize(din.yzw);                     // straight-ray approximation within the cell
    float tauFar = tFar - tNear;                      // remaining arc length in this cell (τ from the entry)

    // plane_intersection_fwd_local (raytrace.py:43) with eye=e3, dir=d3, t_near=0, plane through 0 with normal n̄
    float dp0 = dot(nflat, d3);
    float tauSurf0 = -dot(e3, nflat) / dp0;           // plane n̄·x̄ = 0
    float tq0 = (dp0 >= 0.0) ? 0.0 : max(0.0, tauSurf0);
    vec3 xq0 = e3 + tq0 * d3;
    float invR2 = 1.0 / (R * R);
    int sbase = prim * uK;
    float hsum = 0.0, wsum0 = 0.0;
    for (int s = 0; s < MAX_K; ++s) {
      if (s >= uK) break;
      vec4 so = fetch4(uSiteOff, sbase + s);
      vec3 dv = xq0 - uScale * so.xyz;                // detail sites: exact exp-map images in this frame
      float w = exp(-SOFT_VORONOI_TEMP * dot(dv, dv) * invR2);
      hsum += w * (uScale * so.w);
      wsum0 += w;
    }
    float height = hsum / max(wsum0, 1e-20);

    float dp1 = dp0;
    float tauSurf = (height - dot(e3, nflat)) / dp1;  // plane n̄·x̄ = h
    float tq1 = (dp1 >= 0.0) ? 0.0 : max(0.0, tauSurf);
    vec3 xq1 = e3 + tq1 * d3;
    vec3 csum = vec3(0.0);
    float wsum1 = 0.0;
    for (int s = 0; s < MAX_K; ++s) {
      if (s >= uK) break;
      vec4 so = fetch4(uSiteOff, sbase + s);
      vec3 dv = xq1 - uScale * so.xyz;
      float w = exp(-SOFT_VORONOI_TEMP * dot(dv, dv) * invR2);
      csum += w * fetch4(uSiteRgb, sbase + s).rgb;
      wsum1 += w;
    }
    vec3 colour = csum / max(wsum1, 1e-20);
    // ---- end local shading ----

    float tauNear = 0.0;
    if (dp1 >= 0.0) tauFar = min(tauSurf, tauFar); else tauNear = max(tauSurf, tauNear);  // raytrace.py:225

    float dt = tauFar - tauNear;
    if (dt > 0.0) {
      float delta = -sigma * dt;
      float alpha = 1.0 - exp(delta);
      rgb += colour * alpha * trans;
      logT += delta;
      // G-buffer: record the first segment that takes the transmittance below 0.5 (median depth)
      if (!gotHit && exp(logT) < 0.5) {
        gotHit = true;
        float tX = tNear + tauNear;                       // arc length (since the last hop) of the dense start
        vec4 X = geodesicK(k, o, v, tX);
        vec4 N = translationToK(k, P) * vec4(0.0, nflat); // transported normal (tangent at P ≈ at X)
        oHit = vec4(X.yzw, 1.0);
        oNormal = N;
        oMeta = vec4(tTotal + tX, float(prim), trans, X.x);
      }
    }
    } // shade

    if (crossing) {
      if (hops >= uMaxHops) break;
      hops++;
      tTotal += tDom;
      // transport the ray state through the pairing (points and tangents alike), re-normalise
      mat4 g = uFaceG[exitFace];
      vec4 x = geodesicK(k, o, v, tDom);
      vec4 dv = geodesicDirK(k, o, v, tDom);
      vec4 o2 = projectK(k, g * x);
      vec4 v2 = tangentializeK(k, o2, g * dv);
      // nudge inward so the partner face is not re-detected at t = 0
      o = projectK(k, geodesicK(k, o2, v2, 1e-5));
      v = tangentializeK(k, o, geodesicDirK(k, o2, v2, 1e-5));
      ptNear = 0.0;
      // point location on the partner face: chart lookup, then a short ascent
      vec3 l = logAtOriginK(k, uFaceLocInv[exitFace] * o);
      vec2 uv = clamp(l.xy / uFaceChartHalf[exitFace] * 0.5 + 0.5, 0.0, 0.999);
      ivec2 gij = ivec2(uv * float(LOC_GRID));
      int guess = int(fetchU(uLocGrid, uFacePartner[exitFace] * LOC_GRID * LOC_GRID + gij.y * LOC_GRID + gij.x));
      prim = locateCell(guess, o, kk);
    } else {
      prim = next;
      ptNear = max(ptNear, ptFar);
    }
  }
  float Tfinal = exp(logT);
  rgb += uBackground * Tfinal;
  oColor = vec4(rgb, Tfinal);
}
