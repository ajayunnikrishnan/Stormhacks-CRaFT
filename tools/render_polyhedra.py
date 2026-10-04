"""Render 3/4-view pictures of every fundamental polyhedron for the universe gallery.

Input: web/tools/domains.json (from `node web/tools/dump_domains.mjs`): each face of each domain as
a subdivided geodesic mesh in conformal coordinates (Poincaré ball for H³, stereographic for S³,
metres for E³), so hyperbolic faces bow inward and spherical faces bow outward exactly as the
model shows them. Faces are coloured by pair with the renderer's PAIR_COLOURS (the same tints the
walls get in-world); a label on one face of each pair gives the twist of the gluing.
Output: web/public/thumbs/<id>.png (transparent background).
"""
import json, sys, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "web/tools/domains.json"
OUT = ROOT / "web/public/thumbs"
OUT.mkdir(parents=True, exist_ok=True)
ELEV, AZIM = 24, -56
SIZE_PX = 512

def to_mpl(p):  # scene (x, y up, z) -> mpl (X, Y, Z up), proper rotation about x
    p = np.asarray(p, float)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], axis=-1)

def view_dir():
    e, a = math.radians(ELEV), math.radians(AZIM)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])

def setup(ax, r):
    ax.set_proj_type("persp", focal_length=1.6)
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0, 0))

def shade(base, n, V, L):
    diff = max(0.0, float(n @ L))
    rim = (1 - max(0.0, float(n @ V))) ** 3
    spec = max(0.0, float(n @ ((L + V) / np.linalg.norm(L + V)))) ** 24
    c = np.array(base) * (0.50 + 0.55 * diff) + 0.06 * rim + 0.22 * spec
    return np.clip(c, 0, 1)

def render_domain(dom, pair_cols):
    V = view_dir()
    L = np.array([-0.35, 0.55, 0.75]); L /= np.linalg.norm(L)
    tris, cols, lines, labels = [], [], [], {}
    allpts = []
    for F in dom["faces"]:
        base = pair_cols[F["pair"] % len(pair_cols)]
        centre = to_mpl(F["centre"])
        allpts.append(centre)
        for t in F["tris"]:
            T = to_mpl(np.array(t))
            n = np.cross(T[1] - T[0], T[2] - T[0]); ln = np.linalg.norm(n)
            if ln < 1e-14: continue
            n /= ln
            if n @ (T.mean(0)) < 0: n = -n  # outward
            if n @ V < -0.02: continue       # back-face cull
            tris.append(T); cols.append((*shade(base, n, V, L), 1.0))
        outline = to_mpl(np.array(F["outline"]))
        allpts.extend(outline)
        facing = (centre / (np.linalg.norm(centre) + 1e-12)) @ V
        if facing > 0.05:
            lines.append(outline)
            if F["pair"] not in labels or facing > labels[F["pair"]][0]:
                lab = "mirror" if F["mirrored"] else (f"{round(F['twistDeg'])}°" if F["twistDeg"] > 0.5 else "straight")
                labels[F["pair"]] = (facing, centre, lab)
    r = max(np.linalg.norm(p) for p in allpts) * 0.80
    # one label when every pair has the same twist (e.g. all faces "straight"), else one per pair
    labs = {v[2] for v in labels.values()}
    if len(labs) == 1:
        best = max(labels.values(), key=lambda v: v[0])
        labels = {0: best}
    fig = plt.figure(figsize=(4, 4), dpi=SIZE_PX // 4)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    setup(ax, r)
    pc = Poly3DCollection(tris, facecolors=cols, edgecolors=cols, linewidths=0.4, zsort="average")  # edge = face colour hides AA seams
    ax.add_collection3d(pc)
    lc = Line3DCollection(lines, colors=(1, 1, 1, 0.55), linewidths=1.3)
    ax.add_collection3d(lc)
    for facing, c, lab in labels.values():
        ax.text(c[0] * 1.02, c[1] * 1.02, c[2] * 1.02, lab, color="white", fontsize=8.5, ha="center", va="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.28", fc=(0.03, 0.04, 0.08, 0.75), ec="none"), zorder=10)
    return fig

def render_open():
    fig = plt.figure(figsize=(4, 4), dpi=SIZE_PX // 4)
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    setup(ax, 0.95)
    u = np.linspace(0, 2 * np.pi, 48); v = np.linspace(0, np.pi, 24)
    X = np.outer(np.cos(u), np.sin(v)); Y = np.outer(np.sin(u), np.sin(v)); Z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(X, Y, Z, color=(0.5, 0.65, 1.0, 0.10), linewidth=0, antialiased=True, shade=True)
    ax.plot_wireframe(X, Y, Z, rstride=6, cstride=6, color=(0.6, 0.72, 1.0, 0.35), linewidth=0.8)
    for s in np.linspace(-0.8, 0.8, 5):
        ax.plot([-1, 1], [s, s], [0, 0], color=(0.7, 0.8, 1.0, 0.25), lw=0.8)
        ax.plot([s, s], [-1, 1], [0, 0], color=(0.7, 0.8, 1.0, 0.25), lw=0.8)
    ax.text(0, 0, 0, "∞", color="white", fontsize=28, ha="center", va="center", weight="bold")
    return fig

def main():
    data = json.loads(SRC.read_text())
    pair_cols = data["pairColours"]
    for dom in data["domains"]:
        fig = render_domain(dom, pair_cols)
        fig.savefig(OUT / f"{dom['id']}.png", transparent=True)
        plt.close(fig)
        print("wrote", dom["id"])
    fig = render_open(); fig.savefig(OUT / "none.png", transparent=True); plt.close(fig); print("wrote none")

if __name__ == "__main__":
    main()
