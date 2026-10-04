"""Curved-space embedding of a flat Power Foam scene and the curved power diagram.

Numpy reference for the TS/GLSL geometry (web/src/geometry/space.ts) — the
offline side of docs/WRITEUP.md §3.4–3.6:

  unit model: points x ∈ ℝ⁴ with ⟨x,x⟩_κ = κ (E³: x0 = 1), ⟨x,y⟩_κ = κ x0 y0 + x̄·ȳ
  scale:      a flat scene in metres is scaled by s = √|k| before embedding (Eq. 3.6)
  embedding:  P = cs_κ(|x̄|)·o + sn_κ(|x̄|)·(0, x̂),   R = s·r
  cell rule:  a_i = P_i / cs_κ(R_i);   cell(x) = argmax_i ⟨x, a_i⟩'          (§3.4)
              where ⟨·,·⟩' uses κ' = κ for κ≠0 and κ' = 1 for E³ with a_i = (−pm_i, p̄_i),
              pm = ½(|p̄|² − r²).  This reproduces  H³: min cosh d/cosh r,
              S³: max cos d/cos r,  E³: min |x−p|² − r²  (verified in tests).
  bisector:   ⟨x, a_j − a_i⟩' = 0 — a linear hyperplane ⇒ Eq. (6) applies.
  adjacency:  H³: Euclidean regular triangulation in the Klein model with
              c_i = ā_i/2, w_i = |c_i|² + κ a_i0 (§3.5, Nielsen–Nock);
              S³: edges of the 4D convex hull of {a_i} (normal fan);  E³: flat builder.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from pf_common import FoamScene, edges_to_csr, regular_triangulation_edges


# --------------------------------------------------------------------------- #
# generalised trig / form (mirror of space.ts)
# --------------------------------------------------------------------------- #
def cs(k: int, t):
    return np.cos(t) if k > 0 else (np.cosh(t) if k < 0 else np.ones_like(np.asarray(t, float)))


def sn(k: int, t):
    return np.sin(t) if k > 0 else (np.sinh(t) if k < 0 else np.asarray(t, float))


def form(k: int, x, y):
    """⟨x,y⟩_κ, broadcasting over leading dims."""
    return k * x[..., 0] * y[..., 0] + np.sum(x[..., 1:] * y[..., 1:], axis=-1)


def form_cell(k: int, x, a):
    """⟨x,a⟩' — the cell-rule form (κ' = 1 for E³)."""
    kk = 1 if k == 0 else k
    return kk * x[..., 0] * a[..., 0] + np.sum(x[..., 1:] * a[..., 1:], axis=-1)


def embed(k: int, x3: np.ndarray) -> np.ndarray:
    """Eq. 3.6: flat points (already scaled) -> model points (N,4)."""
    x3 = np.asarray(x3, float)
    n = np.linalg.norm(x3, axis=-1)
    out = np.empty(x3.shape[:-1] + (4,))
    if k == 0:
        out[..., 0] = 1
        out[..., 1:] = x3
        return out
    safe = np.where(n < 1e-300, 1.0, n)
    out[..., 0] = cs(k, n)
    out[..., 1:] = (sn(k, n) / safe)[..., None] * x3
    out[n < 1e-300, 1:] = 0
    return out


def distance(k: int, x, y):
    d = x - y
    if k == 0:
        return np.linalg.norm(d[..., 1:], axis=-1)
    c2 = k * d[..., 0] ** 2 + np.sum(d[..., 1:] ** 2, -1)
    h = np.sqrt(np.maximum(c2, 0)) / 2
    return 2 * np.arcsin(np.minimum(1, h)) if k > 0 else 2 * np.arcsinh(h)


def translation_to(k: int, p: np.ndarray) -> np.ndarray:
    """§3.7 translation isometry o -> p, as a (4,4) matrix acting on column vectors."""
    c = 1.0 if k == 0 else p[0]
    s = np.linalg.norm(p[1:])
    u = p[1:] / s if s > 1e-300 else np.zeros(3)
    M = np.zeros((4, 4))
    M[0, 0] = c
    M[1:, 0] = s * u
    M[0, 1:] = -k * s * u
    M[1:, 1:] = np.eye(3) + (c - 1) * np.outer(u, u)
    return M


# --------------------------------------------------------------------------- #
# curved sites
# --------------------------------------------------------------------------- #
class CurvedSites:
    """Embedded sites for one (κ, s): P (N,4), R (N,), A (N,4) = cell-rule vectors."""

    def __init__(self, k: int, s: float, pos: np.ndarray, radius: np.ndarray, centre: np.ndarray):
        self.k, self.s = k, s
        x = (np.asarray(pos, float) - centre) * s
        self.R = np.asarray(radius, float) * s
        self.P = embed(k, x)
        if k == 0:
            pm = 0.5 * (np.sum(x * x, -1) - self.R**2)
            self.A = np.concatenate([-pm[:, None], x], -1)
        else:
            if k > 0 and np.any(self.R >= np.pi / 2 - 1e-6):
                raise ValueError("S³ requires every radius < π/2: reduce the scale")
            self.A = self.P / cs(k, self.R)[:, None]

    def cell_of(self, x: np.ndarray, chunk: int = 2048) -> np.ndarray:
        """Brute-force cell owner for model points x (M,4): argmax ⟨x,a_i⟩'."""
        out = np.empty(x.shape[0], np.int64)
        kk = 1 if self.k == 0 else self.k
        Aw = self.A.copy()
        Aw[:, 0] *= kk
        for a in range(0, x.shape[0], chunk):
            out[a : a + chunk] = np.argmax(x[a : a + chunk] @ Aw.T, axis=1)
        return out

    def power_value(self, x: np.ndarray, i: np.ndarray) -> np.ndarray:
        """The quantity the cell rule extremises, in its natural form, for checking:
        H³: cosh d/cosh r (min); S³: cos d/cos r (max); E³: |x−p|²−r² (min)."""
        d = distance(self.k, x, self.P[i])
        if self.k == 0:
            return d**2 - self.R[i] ** 2
        return cs(self.k, d) / cs(self.k, self.R[i])


# --------------------------------------------------------------------------- #
# adjacency for one κ
# --------------------------------------------------------------------------- #
def curved_edges(sites: CurvedSites) -> np.ndarray:
    """Undirected edges (i<j) of the curved power diagram's dual."""
    k, A = sites.k, sites.A
    if k == 0:
        return regular_triangulation_edges(A[:, 1:], sites.R**2)
    if k < 0:
        c = A[:, 1:] / 2
        w = np.sum(c * c, -1) + k * A[:, 0]
        return regular_triangulation_edges(c, w)
    # S³: normal fan of conv{a_i}: edges of the 4D hull (Qt: triangulated facets)
    hull = ConvexHull(A, qhull_options="Qt")
    f = hull.simplices  # (F,4)
    e = np.concatenate([f[:, [0, 1]], f[:, [0, 2]], f[:, [0, 3]], f[:, [1, 2]], f[:, [1, 3]], f[:, [2, 3]]], 0)
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def verify_edges(sites: CurvedSites, offsets: np.ndarray, index: np.ndarray, n_samples: int = 20000, seed: int = 0) -> dict:
    """Prompt §3.5 step 4: brute-force check. For random model points x:
    (a) the true owner beats all its neighbours (local optimality at the owner);
    (b) a steepest-ascent walk over the graph from a random start reaches the owner.
    Points are sampled in the model near the sites (S³: uniformly on the whole sphere)."""
    rng = np.random.default_rng(seed)
    k = sites.k
    if k > 0:
        x = rng.standard_normal((n_samples, 4))
        x /= np.linalg.norm(x, axis=-1, keepdims=True)
    else:
        lo, hi = sites.P[:, 1:].min(0), sites.P[:, 1:].max(0)
        if k == 0:
            x = np.concatenate([np.ones((n_samples, 1)), rng.uniform(lo, hi, (n_samples, 3))], -1)
        else:
            xb = rng.uniform(lo, hi, (n_samples, 3))
            x = np.concatenate([np.sqrt(1 + np.sum(xb * xb, -1))[:, None], xb], -1)
    owner = sites.cell_of(x)
    kk = 1 if k == 0 else k
    Aw = sites.A.copy()
    Aw[:, 0] *= kk
    bad = 0
    for i in range(n_samples):
        o = owner[i]
        nb = index[offsets[o] : offsets[o + 1]]
        if np.any(Aw[nb] @ x[i] > Aw[o] @ x[i] + 1e-9):
            bad += 1
    fails = 0
    for i in range(min(n_samples, 2000)):
        c = int(rng.integers(0, sites.P.shape[0]))
        for _ in range(100000):
            nb = index[offsets[c] : offsets[c + 1]]
            if nb.size == 0:
                break
            vals = Aw[nb] @ x[i]
            j = int(np.argmax(vals))
            if vals[j] > Aw[c] @ x[i] + 1e-12:
                c = int(nb[j])
            else:
                break
        if c != owner[i]:
            fails += 1
    return {"samples": n_samples, "local_optimality_violations": int(bad), "ascent_walk_mismatches": int(fails)}


# --------------------------------------------------------------------------- #
# κ sweep -> union adjacency
# --------------------------------------------------------------------------- #
def sweep_values(k_max: float, n_per_sign: int) -> list[float]:
    """Curvatures k (1/m², sign = κ) sampled densely near 0 and up to ±k_max: k = ±k_max·(i/n)²."""
    vals = [k_max * (i / n_per_sign) ** 2 for i in range(1, n_per_sign + 1)]
    return [-v for v in reversed(vals)] + [0.0] + vals


def union_adjacency(scene: FoamScene, centre: np.ndarray, ks: list[float], verify_samples: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Union of per-k edge sets. Returns CSR + per-k statistics."""
    n = scene.n
    all_edges = []
    stats = []
    for kv in ks:
        k = int(np.sign(kv))
        s = float(np.sqrt(abs(kv))) if k != 0 else 1.0
        sites = CurvedSites(k, s, scene.pos, scene.radius, centre)
        e = curved_edges(sites)
        off, idx = edges_to_csr(e, n)
        deg = np.diff(off)
        st = {"k": kv, "kappa": k, "s": s, "n_edges": int(e.shape[0]), "avg_degree": float(deg.mean()), "n_zero_degree": int((deg == 0).sum())}
        if verify_samples:
            st["verify"] = verify_edges(sites, off, idx, verify_samples)
        stats.append(st)
        all_edges.append(e)
    union = np.unique(np.concatenate(all_edges, 0), axis=0)
    off, idx = edges_to_csr(union, n)
    deg = np.diff(off)
    flat_edges = next(st["n_edges"] for st in stats if st["kappa"] == 0)
    info = {
        "samples": stats,
        "union_edges": int(union.shape[0]),
        "union_avg_degree": float(deg.mean()),
        "union_max_degree": int(deg.max()),
        "inflation_vs_flat": float(union.shape[0] / flat_edges),
        "n_zero_degree_union": int((deg == 0).sum()),
    }
    return off, idx, info


# --------------------------------------------------------------------------- #
# exact curved traversal on the CPU (for the union-vs-exact walk test)
# --------------------------------------------------------------------------- #
def plane_exit_after(k: int, A: float, B: float, t_min: float) -> float:
    if k > 0:
        R = np.hypot(A, B)
        if R < 1e-12:
            return np.inf
        base = np.arctan2(B, A) - np.pi / 2
        return base + 2 * np.pi * np.ceil((t_min - base) / (2 * np.pi))
    if k == 0:
        if B <= 0:
            return np.inf
        t = -A / B
        return t if t >= t_min else np.inf
    if abs(B) <= abs(A) or B <= 0:
        return np.inf
    t = 0.5 * np.log((B - A) / (A + B))
    return t if t >= t_min else np.inf


def walk_cells(sites: CurvedSites, offsets: np.ndarray, index: np.ndarray, o: np.ndarray, v: np.ndarray, start: int, t_max: float, max_steps: int = 10000) -> list[int]:
    """Sequence of cells a ray (o,v) traverses, by exact first-exit-face logic over the
    given graph. Faces are the bisectors with neighbours: w = J'(a_j − a_i)."""
    k = sites.k
    kk = 1 if k == 0 else k
    seq = [start]
    prim, t = start, 0.0
    for _ in range(max_steps):
        nb = index[offsets[prim] : offsets[prim + 1]]
        da = sites.A[nb] - sites.A[prim]
        w = da.copy()
        w[:, 0] *= kk
        A = w @ o
        B = w @ v
        best, best_t = -1, np.inf
        for q in range(nb.size):
            te = plane_exit_after(k, A[q], B[q], t)
            if te < best_t:
                best_t, best = te, int(nb[q])
        if best < 0 or best_t > t_max:
            break
        prim, t = best, best_t
        seq.append(prim)
    return seq
