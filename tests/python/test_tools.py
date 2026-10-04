"""Tests for the offline tools (run: .venv/bin/python -m pytest tests/python -q)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import pf_common as pf  # noqa: E402
import synth_scene  # noqa: E402
import export_scene  # noqa: E402
import ref_render  # noqa: E402


@pytest.fixture(scope="module")
def small_scene(tmp_path_factory):
    """A coarse synthetic room (spacing 0.25 -> ~4k cells) exported to a temp dir."""
    d = tmp_path_factory.mktemp("scene")
    ck, _ = synth_scene.make_scene(spacing=0.25, k=4, d=4, seed=1)
    pf.save_checkpoint(d / "ckpt", ck)
    (d / "ckpt" / "cameras.json").write_text(__import__("json").dumps(synth_scene.test_cameras()))
    s, info = export_scene.prepare_scene(d / "ckpt", steiner=True)
    export_scene.write_scene(s, d / "out", info, {"cameras": synth_scene.test_cameras()})
    return d, ck, s, info


def test_softplus_roundtrip():
    y = np.array([1e-4, 0.01, 0.1, 1.0, 300.0])
    assert np.allclose(pf.softplus(pf.inverse_softplus(y)), y, rtol=1e-9, atol=1e-12)


def test_quat_frame_roundtrip():
    rng = np.random.default_rng(0)
    q = rng.standard_normal((500, 4))
    n, t, b = pf.quat_frame(q)
    # orthonormal, right-handed
    assert np.allclose(np.sum(n * t, -1), 0, atol=1e-9)
    assert np.allclose(np.cross(n, t), b, atol=1e-9)
    q2 = pf.frame_to_quat(n, t, b)
    n2, t2, b2 = pf.quat_frame(q2)
    assert np.allclose(n, n2, atol=1e-8) and np.allclose(t, t2, atol=1e-8)


def test_checkpoint_roundtrip(small_scene):
    d, ck, _, _ = small_scene
    ck2 = pf.load_checkpoint(d / "ckpt")
    for k in ("points", "radii", "density", "quaternions", "texel_sites", "texel_sv_axis", "texel_sv_rgb", "texel_height"):
        assert np.array_equal(getattr(ck, k), getattr(ck2, k)), k
    assert ck2.k == 4 and ck2.d == 4


def test_adjacency_matches_brute_force(small_scene):
    """Regular-triangulation walk == exact power-cell ownership (prompt §3.5 step 4)."""
    _, _, s, _ = small_scene
    v = export_scene.verify_adjacency(s, n_samples=5000, seed=3)
    assert v["local_optimality_violations"] == 0
    assert v["descent_walk_mismatches"] == 0


def test_no_zero_degree_cells(small_scene):
    _, _, s, info = small_scene
    assert (np.diff(s.adj_offsets) > 0).all()
    assert info["n_steiner_cells"] > 0


def test_steiner_points_are_in_empty_space(small_scene):
    """Every Steiner sphere lies outside all scene spheres (ratio rule 2 < r̂/r_near < 6, radius 0.8 r̂)."""
    _, ck, _, _ = small_scene
    sc = pf.activate(ck)
    st_pos, st_r = pf.steiner_points(sc.pos, sc.radius, iterations=3, seed=0)
    assert st_pos.shape[0] > 0
    from scipy.spatial import cKDTree

    tree = cKDTree(sc.pos)
    dist, idx = tree.query(st_pos, k=32)
    gap = (dist - sc.radius[idx]).min(axis=1)
    assert (st_r <= 0.8 * gap + 1e-9).all()
    assert (st_r > 0).all()


def test_export_sections_consistent(small_scene):
    d, _, s, _ = small_scene
    ex = ref_render.ExportedScene(d / "out")
    assert ex.n == s.n and ex.k == s.k and ex.d == s.d
    assert np.allclose(ex.pos, s.pos.astype(np.float32))
    assert np.allclose(ex.sigma, s.sigma.astype(np.float16).astype(np.float32))
    assert ex.adj_off[-1] == ex.adj_idx.shape[0]
    assert (ex.adj_idx < ex.n).all()


def test_reference_render_smoke(small_scene):
    """Tiny render: finite, mostly opaque (a closed room), walls reached in bounded steps."""
    d, _, _, _ = small_scene
    ex = ref_render.ExportedScene(d / "out")
    img, steps = ref_render.render(ex, ex.cameras[0], 24, 18)
    assert np.isfinite(img).all()
    assert img.mean() > 0.2
    assert steps.max() < 400 and steps.mean() > 2


def test_sv_colour_formula():
    """SV colour: single axis, direction == axis -> rgb + 0.5 exactly; clamped at 0."""

    class S:  # minimal duck-typed scene
        pass

    s = S()
    s.pos = np.array([[0, 0, 0]], np.float32)
    s.site_off = np.zeros((1, 1, 3), np.float32)
    s.sv_axis = np.array([[[[0, 0, 3.0]]]], np.float32)  # temp 3, axis +z
    s.sv_rgb = np.array([[[[0.2, -0.7, 0.0]]]], np.float32)
    type(s).site_pos = property(lambda self: self.pos[:, None, :] + self.site_off)
    rgb = ref_render.sv_colours(s, np.array([0, 0, -1.0], np.float32))
    assert np.allclose(rgb[0, 0], [0.7, 0.0, 0.5], atol=1e-6)
