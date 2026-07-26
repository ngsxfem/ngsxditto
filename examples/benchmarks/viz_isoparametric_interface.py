"""The isoparametric geometry on a coarse mesh.

Overlays the straight-edged **coarse background mesh** with the **curved
interface** the solver integrates on — the $P^1$ cut of the level set mapped by
the high-order mesh deformation. A zoom panel contrasts the $P^1$ cut (straight
chords) with the isoparametric curve, centred on the point of largest
deformation so the (small, $O(h^2)$) correction is visible.

Inputs are the ``<label>_snap*.vtu`` interface snapshots and the
``<label>_qoi.npz`` (for the background mesh + element count) written by
``ht_convergence.py``.

Usage::

    python viz_isoparametric_interface.py <run_dir> [label] [out.png]

or import :func:`plot_isoparametric`.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv

import viz_common as V


def _nearest(tmap, t):
    f = min(tmap, key=lambda k: abs(tmap[k] - t))
    return f, tmap[f]


def plot_isoparametric(run_dir, label="hicoarse", out="isoparametric_mapping.png",
                       times=(0.0, 0.5, 1.0)):
    npz = np.load(os.path.join(run_dir, f"{label}_qoi.npz"))
    ne, maxh, order = int(npz["ne"]), float(npz["maxh"]), int(npz["order"])
    tri = V.triangulation((npz["verts"], npz["tris"]))
    tmap = V.pvd_times(os.path.join(run_dir, f"{label}_snap.pvd"))

    fig = plt.figure(figsize=(12, 5.2))
    gs = fig.add_gridspec(1, len(times) + 1, width_ratios=[1] * len(times) + [1.15])
    for j, tt in enumerate(times):
        ax = fig.add_subplot(gs[0, j])
        f, tsnap = _nearest(tmap, tt)
        V.draw_mesh(ax, tri)
        for p in V.interface_polylines(f, warp=True):
            ax.plot(p[:, 0], p[:, 1], color="#d62728", lw=2.4)
        ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.5)
        ax.set_title(f"t = {tsnap:.2f}", fontsize=10)
        if j:
            ax.set_yticklabels([])

    # zoom: P1 cut vs isoparametric, centred on the largest deformation
    axz = fig.add_subplot(gs[0, len(times)])
    f, tsnap = _nearest(tmap, times[-1])
    mm = pv.read(f)
    dd = np.asarray(mm.point_data["deform"]); phi = np.asarray(mm.point_data["phi"])
    near = np.abs(phi) < 0.03
    idx = np.where(near)[0][np.argmax(np.linalg.norm(dd[near], axis=1))]
    cx, cy = mm.points[idx, 0] + dd[idx, 0], mm.points[idx, 1] + dd[idx, 1]
    w = 0.06
    V.draw_mesh(axz, tri, lw=1.0)
    for p in V.interface_polylines(f, warp=False):
        axz.plot(p[:, 0], p[:, 1], color="#555555", lw=2.0, ls="--", label="$P^1$ cut")
    for p in V.interface_polylines(f, warp=True):
        axz.plot(p[:, 0], p[:, 1], color="#d62728", lw=2.6, label="isoparametric")
    axz.set_xlim(cx - w, cx + w); axz.set_ylim(cy - w, cy + w)
    axz.set_aspect("equal"); axz.set_title(f"zoom @ t={tsnap:.2f}", fontsize=10)
    h, lab = axz.get_legend_handles_labels()
    seen = dict(zip(lab, h)); axz.legend(seen.values(), seen.keys(), fontsize=8, loc="lower center")

    fig.suptitle(f"Isoparametric geometry on a coarse mesh (order={order}, {ne} elements, "
                 f"h={maxh}): $P^1$ cut + high-order deformation", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "ht_runs"
    label = sys.argv[2] if len(sys.argv) > 2 else "hicoarse"
    out = sys.argv[3] if len(sys.argv) > 3 else "isoparametric_mapping.png"
    plot_isoparametric(run_dir, label, out)
