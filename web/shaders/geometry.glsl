// Unified constant-curvature geometry — GLSL mirror of src/geometry/space.ts.
// Keep function order, names and equation comments identical to the TS file.
// k ∈ {−1, 0, +1}. Points/tangents are vec4 (x0, x1, x2, x3); planes are covectors w
// with plane = { x : dot(w, x) = 0 } (ordinary dot; w = J n for κ≠0, (−c, n̄) for E³).

#define GEOM_EPS 1e-7
#define TWO_PI 6.283185307179586
#define HALF_PI 1.5707963267948966
const vec4 ORIGIN4 = vec4(1.0, 0.0, 0.0, 0.0);

// Eq. (1): ⟨x,y⟩_κ = κ x0 y0 + x1 y1 + x2 y2 + x3 y3
float formK(int k, vec4 x, vec4 y) { return float(k) * x.x * y.x + dot(x.yzw, y.yzw); }

// Generalised trigonometry: sn_κ = sin/sinh, cs_κ = cos/cosh; E³: sn=t, cs=1.
float csK(int k, float t) { return k > 0 ? cos(t) : (k < 0 ? cosh(t) : 1.0); }
float snK(int k, float t) { return k > 0 ? sin(t) : (k < 0 ? sinh(t) : t); }
float acsK(int k, float c) { return k > 0 ? acos(clamp(c, -1.0, 1.0)) : (k < 0 ? acosh(max(1.0, c)) : 0.0); }

// Eq. (2): via chord c² = ⟨x−y,x−y⟩_κ: H³ d = 2 asinh(c/2), S³ d = 2 asin(c/2)
float distK(int k, vec4 x, vec4 y) {
  vec4 d = x - y;
  if (k == 0) return length(d.yzw);
  float c2 = float(k) * d.x * d.x + dot(d.yzw, d.yzw);
  float half_c = sqrt(max(0.0, c2)) * 0.5;
  return k > 0 ? 2.0 * asin(min(1.0, half_c)) : 2.0 * asinh(half_c);
}

// Eq. (3): γ(t) = cs_κ(t)·o' + sn_κ(t)·v
vec4 geodesicK(int k, vec4 o, vec4 v, float t) { return csK(k, t) * o + snK(k, t) * v; }
// Eq. (4): γ'(t) = −κ·sn_κ(t)·o' + cs_κ(t)·v
vec4 geodesicDirK(int k, vec4 o, vec4 v, float t) { return -float(k) * snK(k, t) * o + csK(k, t) * v; }

// Eq. (5): u = (y − cs_κ(d)·x) / sn_κ(d)
vec4 tangentTowardK(int k, vec4 x, vec4 y, out float d) {
  d = distK(k, x, y);
  float s = snK(k, d);
  if (s < GEOM_EPS) return vec4(0.0);
  return (y - csK(k, d) * x) / s;
}

// Eq. (6): f(t) = A cs_κ(t) + B sn_κ(t) = 0, A = w·o', B = w·v. Cell side w·x ≤ 0.
//   E³: t = −A/B, exit ⇔ B > 0
//   H³: t = ½ ln((B−A)/(A+B)), exists ⇔ |B| > |A|, exit ⇔ B > 0
//   S³: exit root t = atan2(B,A) − π/2 (wrapped to [0,2π)); entry root = atan2(B,A) + π/2
bool planeRootK(int k, float A, float B, out float t, out bool isExit) {
  t = 0.0; isExit = false;
  if (k == 0) { if (B == 0.0) return false; t = -A / B; isExit = B > 0.0; return true; }
  if (k < 0) { if (abs(B) <= abs(A)) return false; t = 0.5 * log((B - A) / (A + B)); isExit = B > 0.0; return true; }
  float R = length(vec2(A, B));
  if (R < GEOM_EPS) return false;
  t = atan(B, A) - HALF_PI;
  t -= TWO_PI * floor(t / TWO_PI);
  isExit = true;
  return true;
}
// First EXIT root with t ≥ tMin, or 1e30.
float planeExitAfterK(int k, float A, float B, float tMin) {
  if (k > 0) {
    float R = length(vec2(A, B));
    if (R < GEOM_EPS) return 1e30;
    float base = atan(B, A) - HALF_PI;
    return base + TWO_PI * ceil((tMin - base) / TWO_PI);
  }
  float t; bool ex;
  bool has = planeRootK(k, A, B, t, ex);
  return (has && ex && t >= tMin) ? t : 1e30;
}
// Last ENTRY root with t ≤ tMax, or −1e30.
float planeEntryBeforeK(int k, float A, float B, float tMax) {
  if (k > 0) {
    float R = length(vec2(A, B));
    if (R < GEOM_EPS) return -1e30;
    float base = atan(B, A) + HALF_PI;
    return base + TWO_PI * floor((tMax - base) / TWO_PI);
  }
  float t; bool ex;
  bool has = planeRootK(k, A, B, t, ex);
  return (has && !ex && t <= tMax) ? t : -1e30;
}
float planeValueK(int k, float A, float B, float t) { return A * csK(k, t) + B * snK(k, t); }
// Fused: first EXIT root ≥ tRef (or 1e30) and last ENTRY root ≤ tRef (or −1e30) in one solve.
// Identical results to planeExitAfterK/planeEntryBeforeK; used by the walker's inner loop.
void planeExitEntryK(int k, float A, float B, float tRef, out float tExit, out float tEntry) {
  tExit = 1e30; tEntry = -1e30;
  if (k > 0) {
    if (A * A + B * B < GEOM_EPS * GEOM_EPS) return;
    float phi = atan(B, A);
    float bx = phi - HALF_PI;
    tExit = bx + TWO_PI * ceil((tRef - bx) / TWO_PI);
    float bn = phi + HALF_PI;
    tEntry = bn + TWO_PI * floor((tRef - bn) / TWO_PI);
    return;
  }
  float t;
  if (k == 0) { if (B == 0.0) return; t = -A / B; }
  else { if (abs(B) <= abs(A)) return; t = 0.5 * log((B - A) / (A + B)); }
  if (B > 0.0) { if (t >= tRef) tExit = t; }
  else { if (t <= tRef) tEntry = t; }
}

// Eq. (7): inside ball(p,r) ⇔ A cs_κ(t) + B sn_κ(t) ≥ κ cs_κ(r), A=⟨o',p⟩, B=⟨v,p⟩
//   S³: t ∈ [φ−α, φ+α] + 2πk, α = acos(cos r / R);  H³: E=e^t between roots of
//   (A+B)E² + 2cosh(r)E + (A−B) = 0;  E³: ordinary quadratic.
//   Conditioned form (see space.ts): A = κ + A', A' = −½⟨o'−p,o'−p⟩_κ; H³ roots as E = 1 + δ;
//   S³ via R − cos r computed from small terms and α = 2 asin(√(ε/2)).
float log1pK(float x) { return abs(x) < 1e-4 ? x - 0.5 * x * x + x * x * x / 3.0 : log(1.0 + x); }

bool ballIntervalK(int k, vec4 o, vec4 v, vec4 p, float r, float tMin, out float t1, out float t2) {
  t1 = 0.0; t2 = 0.0;
  if (k == 0) {
    vec3 oc = o.yzw - p.yzw;
    float qb = 2.0 * dot(oc, v.yzw);
    float qc = dot(oc, oc) - r * r;
    float disc = qb * qb - 4.0 * qc;
    if (disc < 0.0) return false;
    float s = sqrt(disc);
    t1 = (-qb - s) * 0.5; t2 = (-qb + s) * 0.5;
    return true;
  }
  vec4 d = o - p;
  float Ap = -0.5 * (float(k) * d.x * d.x + dot(d.yzw, d.yzw));   // A' = A − κ
  float B = formK(k, v, p);
  if (k < 0) {
    float sr = sinh(r), sh = sinh(0.5 * r);
    float disc = sr * sr + 2.0 * Ap - Ap * Ap + B * B;
    if (disc < 0.0) return false;
    float sq = sqrt(disc);
    float den = -1.0 + Ap + B;
    float num0 = -2.0 * sh * sh - Ap - B;
    float ta = log1pK((num0 + sq) / den), tb = log1pK((num0 - sq) / den);
    t1 = min(ta, tb); t2 = max(ta, tb);
    return true;
  }
  float A = 1.0 + Ap;
  float R = length(vec2(A, B));
  float sh = sin(0.5 * r);
  float RmC = (2.0 * Ap + Ap * Ap + B * B) / (R + 1.0) + 2.0 * sh * sh;   // R − cos r
  if (RmC < 0.0) return false;
  float eps = RmC / R;
  float alpha = 2.0 * asin(min(1.0, sqrt(0.5 * eps)));
  float phi = atan(B, A);
  float kk = ceil((tMin - (phi + alpha)) / TWO_PI);
  t1 = phi - alpha + TWO_PI * kk; t2 = phi + alpha + TWO_PI * kk;
  return true;
}

// §3.6: p = cs(|x̄|)·o + sn(|x̄|)·(0, x̂)
vec4 embedPointK(int k, vec3 x) {
  float n = length(x);
  if (k == 0) return vec4(1.0, x);
  if (n < GEOM_EPS) return ORIGIN4;
  return vec4(csK(k, n), (snK(k, n) / n) * x);
}
// log map at the origin: x̄ with |x̄| = d(o,p)
vec3 logAtOriginK(int k, vec4 p) {
  if (k == 0) return p.yzw;
  float d = distK(k, ORIGIN4, p);
  float s = snK(k, d);
  float f = s < GEOM_EPS ? 1.0 : d / s;
  return f * p.yzw;
}
// re-project onto the model (⟨x,x⟩ = κ; E³: x0 = 1)
vec4 projectK(int k, vec4 x) {
  if (k == 0) { x.x = 1.0; return x; }
  return x / sqrt(max(GEOM_EPS, float(k) * formK(k, x, x)));
}

// §3.7: translation isometry o → p = (c, s û):  T = [[c, −κ s ûᵀ],[s û, I + (c−1) û ûᵀ]]
mat4 translationToK(int k, vec4 p) {
  float c = (k == 0) ? 1.0 : p.x;
  float s = length(p.yzw);
  vec3 u = s < GEOM_EPS ? vec3(0.0) : p.yzw / s;
  mat4 m;
  m[0] = vec4(c, s * u);                                   // image of e0
  for (int i = 0; i < 3; ++i) {
    vec3 col = (c - 1.0) * u[i] * u;
    col[i] += 1.0;
    m[i + 1] = vec4(-float(k) * s * u[i], col);            // image of e_i
  }
  return m;
}
// inverse of an isometry: κ≠0: J Mᵀ J; κ=0: [[1,0],[−Rᵀt, Rᵀ]]
mat4 inverseIsometryK(int k, mat4 m) {
  if (k == 0) {
    mat3 R = mat3(m[1].yzw, m[2].yzw, m[3].yzw);
    mat3 Rt = transpose(R);
    vec3 t = -(Rt * m[0].yzw);
    return mat4(vec4(1.0, t), vec4(0.0, Rt[0]), vec4(0.0, Rt[1]), vec4(0.0, Rt[2]));
  }
  mat4 J = mat4(1.0);
  J[0][0] = float(k);
  return J * transpose(m) * J;
}
// make v a unit tangent at x (E³: v0 = 0)
vec4 tangentializeK(int k, vec4 x, vec4 v) {
  if (k == 0) { v.x = 0.0; return vec4(0.0, normalize(v.yzw)); }
  v -= float(k) * formK(k, v, x) * x;
  return v / sqrt(formK(k, v, v));
}
// area of a geodesic sphere of radius R: 4π sn_κ(R)²
float sphereAreaK(int k, float R) { float s = snK(k, R); return 4.0 * 3.141592653589793 * s * s; }

// Eq. (8): E = (Φ/4π)·max(0,⟨u,n⟩)/sn_κ(d)², u = tangent toward L. Returns (direct, antipodal(S³ only), d, 0).
vec4 irradianceK(int k, vec4 x, vec4 n, vec4 L, float power) {
  float d;
  vec4 u = tangentTowardK(k, x, L, d);
  float s = max(1e-3, abs(snK(k, d)));
  float direct = (power / (4.0 * 3.141592653589793)) * max(0.0, formK(k, u, n)) / (s * s);
  float anti = 0.0;
  if (k > 0) {
    float d2 = TWO_PI - d;
    float s2 = max(1e-3, abs(sin(d2)));
    anti = (power / (4.0 * 3.141592653589793)) * max(0.0, -formK(k, u, n)) / (s2 * s2);
  }
  return vec4(direct, anti, d, 0.0);
}
