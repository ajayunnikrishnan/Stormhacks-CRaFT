"""Shared helpers for reading Power Foam checkpoints and building ray-tracing
adjacency on the CPU.

Everything here is a faithful numpy port of code in the Power Foam repo
(`powerfoam/scene.py`, `benchmark.py`). Line references point at the submodule
at `powerfoam/` (commit 9639225). We port rather than import because the repo
modules import NVIDIA Warp at module scope, which this machine cannot run.

Conventions (see docs/ARCHITECTURE.md §2–§3):
  r   = softplus(radii_raw,   beta=100)
  sig = softplus(density_raw, beta=100)
  n, t, b = columns of the rotation matrix of the normalised quaternion (w,x,y,z)
  site_3d[i,j] = p_i + r_i * (u_ij * t_i + v_ij * b_i)
  h[i,j]       = r_i * texel_height[i,j]
  power(x, i)  = |x - p_i|^2 - r_i^2                     # paper Eq. (2)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import ConvexHull, cKDTree

SOFTPLUS_BETA = 100.0


# --------------------------------------------------------------------------- #
# Activations (scene.py: get_radii / get_density use F.softplus(beta=100))
# --------------------------------------------------------------------------- #
def softplus(x: np.ndarray, beta: float = SOFTPLUS_BETA, threshold: float = 20.0) -> np.ndarray:
    """torch.nn.functional.softplus semantics: linear above threshold/beta."""
    x = np.asarray(x, dtype=np.float64)
    out = np.where(x * beta > threshold, x, np.log1p(np.exp(np.minimum(x * beta, threshold))) / beta)
    return out


def inverse_softplus(y: np.ndarray, beta: float = SOFTPLUS_BETA, threshold: float = 20.0) -> np.ndarray:
    """benchmark.py:86 inverse_softplus."""
    y = np.asarray(y, dtype=np.float64)
    mask = y * beta > threshold
    out = np.empty_like(y)
    out[mask] = y[mask]
    out[~mask] = np.log(np.expm1(beta * y[~mask])) / beta
    return out


# --------------------------------------------------------------------------- #
# Quaternion -> (normal, tangent, bitangent)  (scene.py: get_normals/get_tangents)
# --------------------------------------------------------------------------- #
def quat_frame(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.stack([1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1)
    t = np.stack([2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)], -1)
    b = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    t /= np.linalg.norm(t, axis=-1, keepdims=True)
    b /= np.linalg.norm(b, axis=-1, keepdims=True)
    return n, t, b


def frame_to_quat(n: np.ndarray, t: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Inverse of quat_frame. In scene.py the frame vectors are the ROWS of the
    standard rotation matrix R(q): n = R[0,:], t = R[1,:], b = R[2,:]."""
    R = np.stack([n, t, b], axis=-2)  # rows
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.zeros((R.shape[0], 4))
    # Shepperd's method, vectorised with the four branches.
    c0 = tr > 0
    s = np.sqrt(np.maximum(tr, 0) + 1.0) * 2
    q[c0] = np.stack([0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s], -1)[c0]
    c1 = (~c0) & (m00 > m11) & (m00 > m22)
    s = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 1e-12)) * 2
    q[c1] = np.stack([(m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s], -1)[c1]
    c2 = (~c0) & (~c1) & (m11 > m22)
    s = np.sqrt(np.maximum(1.0 + m11 - m00 - m22, 1e-12)) * 2
    q[c2] = np.stack([(m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s], -1)[c2]
    c3 = (~c0) & (~c1) & (~c2)
    s = np.sqrt(np.maximum(1.0 + m22 - m00 - m11, 1e-12)) * 2
    q[c3] = np.stack([(m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s], -1)[c3]
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# --------------------------------------------------------------------------- #
# Checkpoint I/O
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class RawCheckpoint:
    """Raw (pre-activation) tensors exactly as stored in model.pt."""

    points: np.ndarray  # (N,3) f32
    radii: np.ndarray  # (N,)  raw
    density: np.ndarray  # (N,)  raw
    quaternions: np.ndarray  # (N,4)
    texel_sites: np.ndarray  # (N,k,2) units of r
    texel_sv_axis: np.ndarray  # (N,k,3D) raw (norm = temperature)
    texel_sv_rgb: np.ndarray  # (N,k,3D)
    texel_height: np.ndarray  # (N,k) units of r
    config: dict

    @property
    def n(self) -> int:
        return self.points.shape[0]

    @property
    def k(self) -> int:
        return self.texel_sites.shape[1]

    @property
    def d(self) -> int:
        return self.texel_sv_axis.shape[2] // 3


def load_checkpoint(scene_dir: str | Path) -> RawCheckpoint:
    import torch  # local import: only needed to unpickle model.pt

    scene_dir = Path(scene_dir)
    data = torch.load(scene_dir / "model.pt", map_location="cpu", weights_only=True)
    cfg = yaml.safe_load((scene_dir / "config.yaml").read_text()) if (scene_dir / "config.yaml").exists() else {}
    f = lambda key: data[key].float().numpy()
    ck = RawCheckpoint(
        points=f("points"),
        radii=f("radii"),
        density=f("density"),
        quaternions=f("quaternions"),
        texel_sites=f("texel_sites"),
        texel_sv_axis=f("texel_sv_axis"),
        texel_sv_rgb=f("texel_sv_rgb"),
        texel_height=f("texel_height"),
        config=cfg,
    )
    assert ck.texel_sv_axis.shape[2] % 3 == 0
    return ck


def save_checkpoint(scene_dir: str | Path, ck: RawCheckpoint, adjacency=None, adjacency_offsets=None) -> None:
    """Write model.pt + config.yaml in the repo's own format (scene.py:save_pt)."""
    import torch

    scene_dir = Path(scene_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    t = lambda a, dt=torch.float32: torch.from_numpy(np.ascontiguousarray(a)).to(dt)
    n = ck.n
    if adjacency is None:
        adjacency = np.zeros(0, np.int32)
        adjacency_offsets = np.zeros(n + 1, np.int32)
    torch.save(
        {
            "points": t(ck.points),
            "density": t(ck.density),
            "radii": t(ck.radii),
            "quaternions": t(ck.quaternions),
            "texel_sites": t(ck.texel_sites),
            "texel_sv_axis": t(ck.texel_sv_axis),
            "texel_sv_rgb": t(ck.texel_sv_rgb),
            "texel_height": t(ck.texel_height),
            "adjacency": t(adjacency, torch.int32),
            "adjacency_offsets": t(adjacency_offsets, torch.int32),
        },
        scene_dir / "model.pt",
    )
    (scene_dir / "config.yaml").write_text(yaml.safe_dump(ck.config, sort_keys=False))


# --------------------------------------------------------------------------- #
# Activated scene (what the renderer consumes)
# --------------------------------------------------------------------------- #
@dataclasses.dataclass
class FoamScene:
    pos: np.ndarray  # (N,3)
    radius: np.ndarray  # (N,)
    sigma: np.ndarray  # (N,)
    normal: np.ndarray  # (N,3)
    site_off: np.ndarray  # (N,k,3) 3D offset of each detail site from pos
    height: np.ndarray  # (N,k)  displacement along normal (world units)
    sv_axis: np.ndarray  # (N,k,D,3) raw axis (norm = temperature)
    sv_rgb: np.ndarray  # (N,k,D,3)
    adj_offsets: np.ndarray | None = None  # (N+1,) int32 CSR
    adj_index: np.ndarray | None = None  # (E,) int32

    @property
    def n(self):
        return self.pos.shape[0]

    @property
    def k(self):
        return self.site_off.shape[1]

    @property
    def d(self):
        return self.sv_axis.shape[2]

    def sv_temperature(self):
        return np.linalg.norm(self.sv_axis, axis=-1)

    def sv_axis_unit(self):
        return self.sv_axis / np.maximum(self.sv_temperature()[..., None], 1e-20)


def activate(ck: RawCheckpoint) -> FoamScene:
    """scene.py:forward / benchmark.py:test — derive renderable attributes."""
    n, k, d = ck.n, ck.k, ck.d
    r = softplus(ck.radii)
    sig = softplus(ck.density)
    nrm, tan, bit = quat_frame(ck.quaternions.astype(np.float64))
    off = ck.texel_sites * r[:, None, None]  # (N,k,2)
    site_off = off[..., 0:1] * tan[:, None, :] + off[..., 1:2] * bit[:, None, :]
    height = ck.texel_height * r[:, None]
    return FoamScene(
        pos=ck.points.astype(np.float64),
        radius=r,
        sigma=sig,
        normal=nrm,
        site_off=site_off,
        height=height,
        sv_axis=ck.texel_sv_axis.reshape(n, k, d, 3).astype(np.float64),
        sv_rgb=ck.texel_sv_rgb.reshape(n, k, d, 3).astype(np.float64),
    )


# --------------------------------------------------------------------------- #
# Steiner points — port of benchmark.py:94-179 (follows the CODE, not paper Alg. 1)
# --------------------------------------------------------------------------- #
def _greedy_min_overlap_fps(sample_points, sample_radii, threshold, rng):
    """benchmark.py:94 — greedy FPS keeping pairwise overlap <= threshold."""
    n = sample_points.shape[0]
    is_candidate = np.zeros(n, dtype=bool)
    if n == 0:
        return is_candidate
    seed = int(rng.integers(0, n))
    is_candidate[seed] = True
    sum_r = sample_radii + sample_radii[seed]
    dist = np.linalg.norm(sample_points - sample_points[seed], axis=-1)
    max_overlap = (sum_r - dist) / sum_r
    max_overlap[seed] = np.inf
    while True:
        idx = int(np.argmin(max_overlap))
        if max_overlap[idx] > threshold:
            break
        is_candidate[idx] = True
        max_overlap[idx] = np.inf
        sum_r = sample_radii + sample_radii[idx]
        dist = np.linalg.norm(sample_points - sample_points[idx], axis=-1)
        np.maximum(max_overlap, (sum_r - dist) / sum_r, out=max_overlap)
    return is_candidate


def steiner_points(points: np.ndarray, radii: np.ndarray, iterations: int = 10, seed: int = 0, sample_box: tuple | None = None):
    """benchmark.py:137 get_steiner_points. Returns (steiner_pos, steiner_radius_activated).

    Note: the repo returns raw (inverse-softplus) radii because it appends them to
    the model; we return activated radii and let callers decide.
    sample_box=(lo, hi): sample candidates uniformly in that box instead of the repo's
    Normal(mean, 0.5·std) — used to fill a fundamental domain around open scenes (§3.8),
    where the Normal rule leaves the "sky" covered by a few huge, high-degree cells.
    """
    rng = np.random.default_rng(seed)
    points = np.asarray(points, np.float64)
    radii = np.asarray(radii, np.float64)
    mean = points.mean(axis=0, keepdims=True)
    std = points.std(axis=0, ddof=1, keepdims=True)  # torch.std is unbiased
    st_pts = np.empty((0, 3))
    st_r = np.empty((0,))
    for _ in range(iterations):
        all_pts = np.concatenate([points, st_pts], 0)
        all_r = np.concatenate([radii, st_r], 0)
        tree = cKDTree(all_pts)
        m = int(0.25 * points.shape[0])
        if sample_box is None:
            samples = mean + 0.5 * std * rng.standard_normal((m, 3))
        else:
            lo, hi = np.asarray(sample_box[0], float), np.asarray(sample_box[1], float)
            samples = lo + (hi - lo) * rng.uniform(size=(m, 3))
        kq = min(32, all_pts.shape[0])
        dists, idxs = tree.query(samples, k=kq)
        gap = dists - all_r[idxs]  # distance to sphere surface
        closest = np.argmin(gap, axis=1)
        sample_r = gap[np.arange(m), closest]
        closest_idx = idxs[np.arange(m), closest]
        ratio = sample_r / all_r[closest_idx]
        mask = (ratio > 2.0) & (ratio < 6.0)
        sp, sr = samples[mask], sample_r[mask]
        keep = _greedy_min_overlap_fps(sp, sr, 0.1, rng)
        st_pts = np.concatenate([st_pts, sp[keep]], 0)
        st_r = np.concatenate([st_r, 0.8 * sr[keep]], 0)
    return st_pts, st_r


# --------------------------------------------------------------------------- #
# Regular triangulation adjacency — port of benchmark.py:25 build_power_adjacency
# --------------------------------------------------------------------------- #
def regular_triangulation_edges(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Edges (i<j) of the regular triangulation of weighted sites in R^3.

    Lift x -> (x, |x|^2 - w), take the lower convex hull in R^4 (facets with
    negative last normal component), return unique tetra edges. With
    w = r^2 this is the dual of the (unbounded) power diagram.
    """
    pts = np.asarray(points, np.float64)
    lift = np.concatenate([pts, (np.sum(pts * pts, -1) - weights)[:, None]], -1)
    hull = ConvexHull(lift)
    lower = hull.simplices[hull.equations[:, 3] < 0]
    e = np.concatenate(
        [lower[:, [0, 1]], lower[:, [1, 2]], lower[:, [2, 0]], lower[:, [0, 3]], lower[:, [1, 3]], lower[:, [2, 3]]], 0
    )
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


def edges_to_csr(edges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Undirected edge list -> CSR (offsets int32 [n+1], index int32 [2E])."""
    directed = np.concatenate([edges, edges[:, ::-1]], 0)
    order = np.argsort(directed[:, 0], kind="stable")
    directed = directed[order]
    counts = np.bincount(directed[:, 0], minlength=n)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    return offsets, directed[:, 1].astype(np.int32)


def build_power_adjacency(points: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = regular_triangulation_edges(points, np.asarray(radii) ** 2)
    return edges_to_csr(edges, points.shape[0])


# --------------------------------------------------------------------------- #
# Morton sort — port of geometry.py:morton_code_kernel (21 bits per axis)
# --------------------------------------------------------------------------- #
def _expand_bits(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.uint64) & np.uint64(0x1FFFFF)
    v = (v | (v << np.uint64(32))) & np.uint64(0x001F00000000FFFF)
    v = (v | (v << np.uint64(16))) & np.uint64(0x001F0000FF0000FF)
    v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
    return v


def morton_order(points: np.ndarray) -> np.ndarray:
    lo, hi = points.min(0), points.max(0)
    dims = np.maximum(hi - lo, 1e-6)
    nrm = (points - lo) / dims
    res = 2097152.0
    q = np.clip(nrm * res, 0, res - 1).astype(np.uint64)
    code = _expand_bits(q[:, 0]) * np.uint64(4) + _expand_bits(q[:, 1]) * np.uint64(2) + _expand_bits(q[:, 2])
    return np.argsort(code, kind="stable")


def permute_scene(s: FoamScene, perm: np.ndarray) -> FoamScene:
    return FoamScene(
        pos=s.pos[perm],
        radius=s.radius[perm],
        sigma=s.sigma[perm],
        normal=s.normal[perm],
        site_off=s.site_off[perm],
        height=s.height[perm],
        sv_axis=s.sv_axis[perm],
        sv_rgb=s.sv_rgb[perm],
    )


def append_steiner(s: FoamScene, st_pos: np.ndarray, st_r: np.ndarray) -> FoamScene:
    """benchmark.py:182 add_steiner_points: sigma≈0 (raw −10), zero texels, zero rgb.

    Steiner cells never shade (sigma < 1e-3 short-circuits the walk), so their
    attribute values are irrelevant; we keep them finite and tiny.
    """
    m = st_pos.shape[0]
    k, d = s.k, s.d
    rng = np.random.default_rng(1)
    q = rng.standard_normal((m, 4))
    nrm, _, _ = quat_frame(q)
    return FoamScene(
        pos=np.concatenate([s.pos, st_pos], 0),
        radius=np.concatenate([s.radius, st_r], 0),
        sigma=np.concatenate([s.sigma, np.full(m, float(softplus(np.array(-10.0))))], 0),
        normal=np.concatenate([s.normal, nrm], 0),
        site_off=np.concatenate([s.site_off, np.zeros((m, k, 3))], 0),
        height=np.concatenate([s.height, np.zeros((m, k))], 0),
        sv_axis=np.concatenate([s.sv_axis, np.tile(np.array([0.0, 0.0, 1.0]), (m, k, d, 1))], 0),
        sv_rgb=np.concatenate([s.sv_rgb, np.zeros((m, k, d, 3))], 0),
    )


def start_cell(s: FoamScene, eye: np.ndarray) -> int:
    """benchmark.py:324 — argmin of power distance |p-eye|^2 - r^2 (brute force)."""
    d2 = np.sum((s.pos - eye[None, :]) ** 2, -1)
    return int(np.argmin(d2 - s.radius**2))


def brute_force_power_cell(s: FoamScene, x: np.ndarray) -> np.ndarray:
    """For sample points x (M,3): index of the power cell containing each (brute force)."""
    tree = cKDTree(s.pos)
    kq = min(256, s.n)
    _, idx = tree.query(x, k=kq)  # candidates; power cell owner is usually among nearest
    cand = s.pos[idx]  # (M,kq,3)
    pw = np.sum((cand - x[:, None, :]) ** 2, -1) - s.radius[idx] ** 2
    return idx[np.arange(x.shape[0]), np.argmin(pw, axis=1)]
