"""Tests for the curved power diagram (docs/WRITEUP.md §3.4–3.5)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import pf_common as pf  # noqa: E402
import synth_scene  # noqa: E402
import export_scene  # noqa: E402
import curved as cv  # noqa: E402


@pytest.fixture(scope="module")
def scene():
    """Coarse synthetic room with Steiner points, Morton sorted (no adjacency yet)."""
    ck, _ = synth_scene.make_scene(spacing=0.3, k=4, d=4, seed=2)
    s = pf.activate(ck)
    st_pos, st_r = pf.steiner_points(s.pos, s.radius, iterations=4, seed=0)
    s = pf.append_steiner(s, st_pos, st_r)
    s = pf.permute_scene(s, pf.morton_order(s.pos))
    centre = 0.5 * (s.pos.min(0) + s.pos.max(0))
    return s, centre


def test_flat_limit_of_curved_power_distance():
    """§3.4: cosh d/cosh r ≈ 1 + (d²−r²)/2 and cos d/cos r ≈ 1 − (d²−r²)/2 for small d, r,
    so both orderings reduce to the Euclidean power distance |x−p|² − r² (paper Eq. 2)."""
    rng = np.random.default_rng(0)
    for s in [0.1, 0.03, 0.01]:
        d = rng.uniform(0, 1, 1000) * s
        r = rng.uniform(0, 1, 1000) * s
        pw = d * d - r * r
        err_h = np.abs((np.cosh(d) / np.cosh(r)) - (1 + pw / 2))
        err_s = np.abs((np.cos(d) / np.cos(r)) - (1 - pw / 2))
        assert err_h.max() < 2 * s**4 and err_s.max() < 2 * s**4
    # and the ordering agrees with the flat one for a random set of sites at small scale
    pos = rng.uniform(-1, 1, (300, 3))
    rad = rng.uniform(0.05, 0.3, 300)
    x = rng.uniform(-1, 1, (2000, 3))
    flat_owner = np.argmin(np.sum((x[:, None] - pos[None]) ** 2, -1) - rad[None] ** 2, axis=1)
    for k in (-1, 1):
        sites = cv.CurvedSites(k, 0.02, pos, rad, np.zeros(3))
        owner = sites.cell_of(cv.embed(k, x * 0.02))
        assert np.mean(owner == flat_owner) > 0.995  # ties / near-ties aside


def test_cell_rule_matches_natural_form():
    """argmax ⟨x,a_i⟩' equals  H³: argmin cosh d/cosh r,  S³: argmax cos d/cos r,  E³: argmin |x−p|²−r²."""
    rng = np.random.default_rng(1)
    pos = rng.uniform(-1, 1, (200, 3))
    rad = rng.uniform(0.05, 0.4, 200)
    for k, s in ((-1, 0.7), (0, 1.0), (1, 0.5)):
        sites = cv.CurvedSites(k, s, pos, rad, np.zeros(3))
        if k > 0:
            x = rng.standard_normal((3000, 4))
            x /= np.linalg.norm(x, axis=-1, keepdims=True)
        else:
            x = cv.embed(k, rng.uniform(-1.2, 1.2, (3000, 3)) * s)
        owner = sites.cell_of(x)
        allv = np.stack([sites.power_value(x, np.full(x.shape[0], i)) for i in range(sites.P.shape[0])], 1)
        expected = np.argmax(allv, 1) if k > 0 else np.argmin(allv, 1)
        assert np.mean(owner == expected) > 0.999


def test_bisector_is_linear_and_through_sphere_intersection():
    """Points equidistant in the power sense satisfy ⟨x, a_i − a_j⟩' = 0; a point on both
    spheres (d_i = r_i, d_j = r_j) lies on the bisector."""
    for k in (-1, 1):
        s = 0.5
        pos = np.array([[0.0, 0, 0], [0.6, 0.1, 0]])
        rad = np.array([0.4, 0.35])
        sites = cv.CurvedSites(k, s, pos, rad, np.zeros(3))
        # find a point on both spheres by solving in the plane y: x = (t, u, 0)
        from scipy.optimize import fsolve

        def f(z):
            x = cv.embed(k, np.array([z[0], z[1], 0.0]))
            return [cv.distance(k, x, sites.P[0]) - sites.R[0], cv.distance(k, x, sites.P[1]) - sites.R[1]]

        z = fsolve(f, [0.15, 0.1])
        x = cv.embed(k, np.array([z[0], z[1], 0.0]))
        assert abs(cv.form_cell(k, x, sites.A[0]) - cv.form_cell(k, x, sites.A[1])) < 1e-8


@pytest.mark.parametrize("kv", [-1.0, -0.2, 0.0, 0.2, 0.9])
def test_adjacency_matches_brute_force(scene, kv):
    """§3.5 step 4 for H³ (Klein), E³ and S³ (4D hull): local optimality + ascent walks."""
    s, centre = scene
    k = int(np.sign(kv))
    sc = float(np.sqrt(abs(kv))) if k else 1.0
    # keep the S³ scene inside a hemisphere with radii < π/2
    if k > 0:
        ext = np.linalg.norm(s.pos - centre, axis=-1).max()
        sc = min(sc, 1.2 / ext)
    sites = cv.CurvedSites(k, sc, s.pos, s.radius, centre)
    e = cv.curved_edges(sites)
    off, idx = pf.edges_to_csr(e, s.n)
    v = cv.verify_edges(sites, off, idx, n_samples=4000, seed=1)
    assert v["local_optimality_violations"] == 0
    assert v["ascent_walk_mismatches"] == 0


def test_union_graph_walk_matches_exact_walk(scene):
    """A superset of the true adjacency gives the same first-exit walk (§3.5). Also at k
    values NOT in the sweep, to check the union covers the combinatorics in between."""
    s, centre = scene
    ext = np.linalg.norm(s.pos - centre, axis=-1).max()
    k_max = (1.2 / ext) ** 2  # S³ hemisphere limit
    ks = cv.sweep_values(k_max, 6)
    off_u, idx_u, info = cv.union_adjacency(s, centre, ks)
    assert info["n_zero_degree_union"] == 0
    assert info["inflation_vs_flat"] < 3.0
    rng = np.random.default_rng(3)
    for kv in [-0.37 * k_max, 0.0, 0.53 * k_max, 0.95 * k_max]:  # not in the sweep
        k = int(np.sign(kv))
        sc = float(np.sqrt(abs(kv))) if k else 1.0
        sites = cv.CurvedSites(k, sc, s.pos, s.radius, centre)
        off_e, idx_e = pf.edges_to_csr(cv.curved_edges(sites), s.n)
        mism = 0
        for _ in range(60):
            o3 = centre + rng.uniform(-1.5, 1.5, 3) * np.array([1, 0.6, 1])
            o = cv.embed(k, (o3 - centre) * sc)
            d = rng.standard_normal(3)
            d /= np.linalg.norm(d)
            v = np.concatenate([[0.0], d])
            if k != 0:  # tangentialise at o
                v = v - k * cv.form(k, v, o) * o
                v = v / np.sqrt(cv.form(k, v, v))
            start = int(sites.cell_of(o[None])[0])
            t_max = 2.5 * sc if k != 0 else 2.5
            a = cv.walk_cells(sites, off_e, idx_e, o, v, start, t_max)
            b = cv.walk_cells(sites, off_u, idx_u, o, v, start, t_max)
            if a != b:
                mism += 1
        assert mism == 0, f"k={kv}: {mism} walks differ between exact and union graphs"


def test_translation_moves_origin_and_preserves_form():
    rng = np.random.default_rng(4)
    for k in (-1, 0, 1):
        p = cv.embed(k, rng.uniform(-0.5, 0.5, 3))
        M = cv.translation_to(k, p)
        o = np.array([1.0, 0, 0, 0])
        assert np.allclose(M @ o, p)
        J = np.diag([k, 1, 1, 1.0])
        if k != 0:
            assert np.allclose(M.T @ J @ M, J, atol=1e-12)
        else:
            assert np.allclose(M[0], [1, 0, 0, 0]) and np.allclose(M[1:, 1:], np.eye(3))
