"""Storyboard and animation of the rising bubble.

Draws the curved (isoparametric) interface over the coarse background mesh across
time, as a storyboard (a row of frames) and an animated GIF. Reads the
``<label>_snap*.vtu`` snapshots and the ``<label>_qoi.npz`` (for the mesh) written
by the drivers.

The bubble is shaded by filling **only the main closed interface loop** — small
spurious contour fragments (which can appear on a coarse cut) are outlined but not
filled, so no unphysical chord is ever drawn across the bubble.

Usage::

    python viz_animation.py <run_dir> [label] [out_prefix]

or import :func:`plot_storyboard` / :func:`make_gif`.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import viz_common as V


def _frame(ax, tri, vtu, ymax=2.0, lw=1.1):
    V.draw_mesh(ax, tri, color="#c4cad4", lw=0.5)
    polys = V.interface_polylines(vtu, warp=True)
    for p in polys:
        ax.plot(p[:, 0], p[:, 1], color="#d62728", lw=lw, solid_capstyle="round")
    # shade only the largest loop, explicitly closed -> never a chord across the bubble
    if polys:
        main = max(polys, key=len)
        closed = np.vstack([main, main[:1]]) if not np.allclose(main[0], main[-1]) else main
        ax.fill(closed[:, 0], closed[:, 1], color="#d62728", alpha=0.10)
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, ymax)


def _mesh_and_times(run_dir, label):
    d = np.load(os.path.join(run_dir, f"{label}_qoi.npz"))
    tri = V.triangulation((d["verts"], d["tris"]))
    tmap = V.pvd_times(os.path.join(run_dir, f"{label}_snap.pvd"))
    return tri, tmap, int(d["ne"]), int(d["order"])


def plot_storyboard(run_dir, label="re_h08", out="storyboard.png",
                    times=(0.0, 0.6, 1.2, 1.8, 2.4, 3.0)):
    tri, tmap, ne, order = _mesh_and_times(run_dir, label)
    fig, axes = plt.subplots(1, len(times), figsize=(2.0 * len(times), 5.0))
    for ax, tt in zip(axes, times):
        f = min(tmap, key=lambda k: abs(tmap[k] - tt))
        _frame(ax, tri, f)
        ax.set_title(f"t = {tmap[f]:.2f}", fontsize=9)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    fig.suptitle(f"Rising bubble (coarse mesh, order={order}, {ne} elements)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def make_gif(run_dir, label="re_h08", out="rising_bubble.gif", dpi=170, lw=1.1):
    """Animate the interface.

    ``dpi`` and ``lw`` are deliberately not at their old values (90 / 2.2): at
    90 dpi the domain is only ~270 px wide, so a 2.2 pt interface line covers
    about 0.02 in domain units -- wider than the interface features one is
    trying to show, which turns ordinary discretisation error into what looks
    like an oscillating interface. Drawing the interface thinner than the
    detail it carries is the point. This only helps if the snapshots were
    written with enough subdivision (see ``run_case``); otherwise the contour
    is one straight chord per cut element no matter how it is drawn."""
    tri, tmap, ne, order = _mesh_and_times(run_dir, label)
    tmp = os.path.join(os.path.dirname(out) or ".", "_frames"); os.makedirs(tmp, exist_ok=True)
    frames = []
    for k, tt in enumerate(sorted(tmap.values())):
        f = min(tmap, key=lambda kk: abs(tmap[kk] - tt))
        fig, ax = plt.subplots(figsize=(3.0, 5.2))
        _frame(ax, tri, f, lw=lw)
        ax.set_title(f"t = {tt:.2f}", fontsize=9)
        fig.tight_layout(); p = os.path.join(tmp, f"f{k:03d}.png")
        fig.savefig(p, dpi=dpi); plt.close(fig); frames.append(imageio.imread(p))
    imageio.mimsave(out, frames, duration=0.12, loop=0)
    print(f"wrote {out} ({len(frames)} frames)")
    return out


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "ht_runs"
    label = sys.argv[2] if len(sys.argv) > 2 else "re_h08"
    prefix = sys.argv[3] if len(sys.argv) > 3 else ""
    plot_storyboard(run_dir, label, (prefix or "") + "storyboard.png")
    make_gif(run_dir, label, (prefix or "") + "rising_bubble.gif")
