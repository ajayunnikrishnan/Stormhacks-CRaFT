"""§3.11: measure the locally-flat shading approximation on the exported scene.

The curved walker shades a cell by mapping the ray into the cell frame M_p⁻¹, taking
the log map, and treating the ray as a straight line there. The undisplaced dipole
plane {⟨x,N⟩_κ = 0} is exactly linear in these coordinates, so the only error is the
straight-ray assumption. We compare, for random rays through shaded cells, the
locally-flat crossing parameter against the exact Eq. (6) crossing of the same plane and
report max |Δt| / R over cells. The bound to document: |Δt| / R ≲ C·|κ|·R² with C = O(1).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import curved as cv  # noqa: E402
import ref_render_curved as rc  # noqa: E402

SCENE = ROOT / "web/public/scenes/synth_room"


@pytest.mark.skipif(not (SCENE / "scene.json").exists(), reason="export the synth scene first")
@pytest.mark.parametrize("kv", [-0.12, 0.03])
def test_local_flat_shading_error(kv):
    ex = rc.load(SCENE)
    ref = rc.CurvedRef(ex, kv)
    k, s = ref.k, ref.s
    rng = np.random.default_rng(0)
    shaded = np.flatnonzero(ex.sigma > 1e-3)
    cells = rng.choice(shaded, size=400, replace=False)
    worst = 0.0
    worst_bound_ratio = 0.0
    n_tested = 0
    for i in cells:
        P, R, n = ref.P[i], ref.R[i], ex.normal[i].astype(np.float64)
        M, Minv = ref.M[i], ref.Minv[i]
        N = M @ np.concatenate([[0.0], n])  # transported normal: tangent at P
        w = N.copy()
        w[0] *= k  # covector J N (κ≠0)
        for _ in range(5):
            # random ray entering the ball of cell i: origin on the ball surface, direction inward-ish
            dir0 = rng.standard_normal(3)
            dir0 /= np.linalg.norm(dir0)
            start_local = cv.embed(k, R * dir0)  # a point on the geodesic sphere of radius R about P (in the frame)
            o = M @ start_local
            v = np.concatenate([[0.0], rng.standard_normal(3)])
            v = v - k * cv.form(k, v, o) * o
            v /= np.sqrt(cv.form(k, v, v))
            # make it head inward: toward P
            toP, _ = rc.tangent_toward(k, o, P)
            if cv.form(k, v, toP) < 0.3:
                v = toP
            # exact crossing of the undisplaced plane (Eq. 6), first root ≥ 0
            A, B = np.dot(w, o), np.dot(w, v)
            te = cv.plane_exit_after(k, A, B, 0.0)
            tn = rc.plane_entry_before(k, A, B, 2 * R + 1e-9) if k <= 0 else np.inf
            cand = [t for t in (te, tn) if np.isfinite(t) and 0 <= t <= 2 * R]
            if k > 0:
                # S³: get both roots in [0, 2R]
                Rr = np.hypot(A, B)
                if Rr < 1e-12:
                    continue
                base = np.arctan2(B, A)
                cand = [t for t in (base - np.pi / 2, base + np.pi / 2, base + 3 * np.pi / 2) if 0 <= t <= 2 * R]
            if not cand:
                continue
            t_exact = min(cand)
            # locally flat crossing: e3 + τ d3 on n̄·x̄ = 0, with (e3, d3) from the entry point
            xin = Minv @ o
            din = Minv @ v
            e3 = rc.log_at_origin(k, xin)
            d3 = din[1:] / np.linalg.norm(din[1:])
            dp = np.dot(n, d3)
            if abs(dp) < 1e-6:
                continue
            tau = -np.dot(e3, n) / dp
            err = abs(tau - t_exact)
            n_tested += 1
            worst = max(worst, err / R)
            worst_bound_ratio = max(worst_bound_ratio, err / R / (abs(kv) * 0 + R * R))  # err/R relative to R² (κ=±1 in unit model)
    assert n_tested > 500
    print(f"\nk={kv}: κ={k} s={s:.3f}  max |Δt|/R = {worst:.2e}  (max R = {ref.R[shaded].max():.4f}, R² = {ref.R[shaded].max()**2:.2e}), ratio to R² = {worst_bound_ratio:.2f}")
    # error is O(|κ| R²) relative to R: allow a constant of 4
    assert worst < 4.0 * ref.R[shaded].max() ** 2


def test_write_report(tmp_path):
    """Writes the measured numbers for docs/WRITEUP.md (no assertion)."""
    if not (SCENE / "scene.json").exists():
        pytest.skip("no scene")
    out = {}
    for kv in (-0.12, -0.03, 0.01, 0.03):
        ex = rc.load(SCENE)
        ref = rc.CurvedRef(ex, kv)
        shaded = np.flatnonzero(ex.sigma > 1e-3)
        out[str(kv)] = {"kappa": ref.k, "s": ref.s, "max_R": float(ref.R[shaded].max()), "kappa_R2": float(ref.R[shaded].max() ** 2)}
    (ROOT / "tests/out").mkdir(exist_ok=True)
    (ROOT / "tests/out/local_flat_bound.json").write_text(json.dumps(out, indent=1))
