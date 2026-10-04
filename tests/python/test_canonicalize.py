"""The similarity transform must commute with activation: activating the transformed checkpoint
gives the transformed activated scene (positions, radii, densities, frames, site offsets, SV axes)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from canonicalize_scene import rotation_from_to, rot_y, transform_checkpoint, transform_cameras, crop  # noqa: E402
from pf_common import RawCheckpoint, activate, inverse_softplus  # noqa: E402


def _random_ck(n=200, k=3, d=2, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    return RawCheckpoint(
        points=rng.normal(size=(n, 3)).astype(np.float32) * 3,
        radii=inverse_softplus(rng.uniform(0.05, 0.5, n)).astype(np.float32),
        density=inverse_softplus(rng.uniform(0.1, 50, n)).astype(np.float32),
        quaternions=q.astype(np.float32),
        texel_sites=rng.normal(size=(n, k, 2)).astype(np.float32) * 0.3,
        texel_sv_axis=rng.normal(size=(n, k, 3 * d)).astype(np.float32),
        texel_sv_rgb=rng.uniform(size=(n, k, 3 * d)).astype(np.float32),
        texel_height=rng.normal(size=(n, k)).astype(np.float32) * 0.1,
        config={"x": 1},
    )


def test_rotation_from_to():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = rng.normal(size=3); b = rng.normal(size=3)
        R = rotation_from_to(a, b)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(R), 1)
        assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-12)
    assert np.allclose(rotation_from_to([0, 0, 1], [0, 0, -1]) @ [0, 0, 1], [0, 0, -1])


def test_transform_commutes_with_activation():
    ck = _random_ck()
    R = rot_y(0.7) @ rotation_from_to([0.2, 0.9, -0.3], [0, 1, 0])
    s, c = 2.5, np.array([0.3, -1.2, 0.8])
    A = activate(ck)
    B = activate(transform_checkpoint(ck, R, s, c))
    assert np.allclose(B.pos, s * (A.pos @ R.T - c), atol=1e-4)
    assert np.allclose(B.radius, s * A.radius, rtol=1e-4)
    assert np.allclose(B.sigma, A.sigma / s, rtol=1e-3)
    # frames rotate (sign of the quaternion is irrelevant: compare the frame itself)
    assert np.allclose(B.normal, A.normal @ R.T, atol=1e-4)
    assert np.allclose(B.site_off, s * (A.site_off @ R.T), atol=1e-4)
    assert np.allclose(B.height, s * A.height, rtol=1e-4)
    assert np.allclose(B.sv_axis, A.sv_axis @ R.T, atol=1e-4)
    assert np.allclose(B.sv_rgb, A.sv_rgb)
    # optical depth across a cell is invariant: sigma·r
    assert np.allclose(B.sigma * B.radius, A.sigma * A.radius, rtol=1e-3)


def test_cameras_and_crop():
    R = rot_y(1.0); s, c = 0.5, np.array([1.0, 2.0, 3.0])
    cams = [{"eye": [1, 2, 3], "right": [1, 0, 0], "up": [0, 1, 0]}]
    out = transform_cameras(cams, R, s, R @ np.array([1.0, 2.0, 3.0]))  # centre ON the eye → eye maps to the origin
    assert np.allclose(out[0]["eye"], [0, 0, 0], atol=1e-12) and np.allclose(np.linalg.norm(out[0]["right"]), 1)
    assert np.allclose(transform_cameras(cams, R, s, c)[0]["eye"], s * (R @ [1, 2, 3] - c))
    ck = _random_ck()
    ck2, idx = crop(ck, radius=2.0, below=1.0, above=1.0)
    P = ck2.points
    assert ck2.n == idx.size and (np.hypot(P[:, 0], P[:, 2]) <= 2.0).all() and (P[:, 1] >= -1).all() and (P[:, 1] <= 1).all()
    assert 0 < ck2.n < ck.n
