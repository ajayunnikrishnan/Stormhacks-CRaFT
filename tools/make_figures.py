"""Generate the validation figures for docs/WRITEUP.md into docs/figures/ (matplotlib).

  falloff.png        irradiance vs distance for κ = −1, 0, +1 (Eq. 8) against 1/d²
  angle_excess.png   triangle angle sum vs side length in the three geometries (Gauss–Bonnet)
  flat_limit.png     |cosh d/cosh r − (1 + (d²−r²)/2)| and the S³ analogue vs scale (O(s⁴))
  psnr.png           GPU fp32 vs references (numbers from PROGRESS.md measurements)
  cube_sizes.png     dihedral angle of a cube vs face distance h in H³ and S³, marking 72° / 120°

Usage: python tools/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def sn(k, t):
    return np.sin(t) if k > 0 else (np.sinh(t) if k < 0 else t)


def falloff():
    d = np.linspace(0.2, 2.8, 300)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, lab, c in [(-1, "H³  Φ/4π sinh²d", "#2a5bd7"), (0, "E³  Φ/4π d²", "#4b8f3a"), (1, "S³  Φ/4π sin²d", "#d7742a")]:
        ax.plot(d, 1 / (4 * np.pi * sn(k, d) ** 2), label=lab, color=c, lw=2)
    ax.set_yscale("log")
    ax.set_xlabel("geodesic distance d (unit curvature)")
    ax.set_ylabel("irradiance E / Φ")
    ax.set_title("Eq. (8): point-light falloff — the light meter's three curves")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "falloff.png", dpi=150)


def angle_excess():
    # equilateral triangles of side a: cos A = (cos a − cos² a)/sin² a (S³), cosh analogue (H³)
    a = np.linspace(0.05, 2.0, 300)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    with np.errstate(invalid="ignore"):
        A_s = np.arccos((np.cos(a) - np.cos(a) ** 2) / np.sin(a) ** 2)
        A_h = np.arccos((np.cosh(a) ** 2 - np.cosh(a)) / np.sinh(a) ** 2)
    ax.plot(a, np.degrees(3 * A_h), color="#2a5bd7", lw=2, label="H³")
    ax.axhline(180, color="#4b8f3a", lw=2, label="E³")
    ax.plot(a, np.degrees(3 * A_s), color="#d7742a", lw=2, label="S³")
    ax.set_xlabel("side length a of an equilateral geodesic triangle (unit curvature)")
    ax.set_ylabel("angle sum (°)")
    ax.set_title("Gauss–Bonnet: Σ angles − 180° = κ · area (what the beacons show)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "angle_excess.png", dpi=150)


def flat_limit():
    s = np.logspace(-2.5, -0.3, 60)
    rng = np.random.default_rng(0)
    d = rng.uniform(0, 1, 2000)
    r = rng.uniform(0, 1, 2000)
    eh, es = [], []
    for si in s:
        dd, rr = d * si, r * si
        pw = dd * dd - rr * rr
        eh.append(np.abs(np.cosh(dd) / np.cosh(rr) - (1 + pw / 2)).max())
        es.append(np.abs(np.cos(dd) / np.cos(rr) - (1 - pw / 2)).max())
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.loglog(s, eh, color="#2a5bd7", lw=2, label="H³: |cosh d/cosh r − (1 + (d²−r²)/2)|")
    ax.loglog(s, es, color="#d7742a", lw=2, label="S³: |cos d/cos r − (1 − (d²−r²)/2)|")
    ax.loglog(s, 2 * s**4, "k--", lw=1, label="2·s⁴")
    ax.set_xlabel("scale s = √|k| (d, r ≤ s)")
    ax.set_ylabel("max deviation from the Euclidean power distance")
    ax.set_title("§3.4 flat limit of the curved power-cell rule")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "flat_limit.png", dpi=150)


def psnr():
    labels = ["flat GPU vs\nnumpy port", "curved κ=0\nvs flat", "k=1e-6\nvs flat", "k=1e-3\nvs fp64", "k=+0.03 (S³)\nvs fp64", "k=−0.12 (H³)\nvs fp64"]
    vals = [72.1, 82.3, 51.8, 66.2, 46.8, 45.6]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(labels, vals, color=["#4b8f3a", "#4b8f3a", "#888", "#2a5bd7", "#d7742a", "#2a5bd7"])
    ax.axhline(40, color="r", ls="--", lw=1, label="target 40 dB")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Renderer validation (synthetic scene, camera 'corner')")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 95)
    fig.tight_layout()
    fig.savefig(OUT / "psnr.png", dpi=150)


def cube_sizes():
    h = np.linspace(0.01, 1.2, 300)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(h, np.degrees(np.arccos(np.clip(np.sinh(h) ** 2, -1, 1))), color="#2a5bd7", lw=2, label="H³: cos φ = sinh²h")
    hs = np.linspace(0.01, np.pi / 2 - 0.01, 300)
    ax.plot(hs, np.degrees(np.arccos(np.clip(-np.sin(hs) ** 2, -1, 1))), color="#d7742a", lw=2, label="S³: cos φ = −sin²h")
    ax.axhline(90, color="#4b8f3a", lw=1.5, label="E³: 90° for any h")
    for y, x, t in [(72, 0.5306, "{4,3,5}: h = 0.531"), (120, np.pi / 4, "{4,3,3}: h = π/4")]:
        ax.plot([x], [y], "ko")
        ax.annotate(t, (x, y), textcoords="offset points", xytext=(8, -12), fontsize=9)
    ax.set_xlabel("face distance h (inradius) of a regular cube")
    ax.set_ylabel("dihedral angle φ (°)")
    ax.set_title("§3.8 cube sizes from dihedral angles")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cube_sizes.png", dpi=150)


if __name__ == "__main__":
    falloff()
    angle_excess()
    flat_limit()
    psnr()
    cube_sizes()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
