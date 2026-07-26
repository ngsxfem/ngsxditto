"""Colour the interface by the mean curvature |H|.

The ``ht_redist_probe.py`` driver dumps the mean curvature ``H`` (from
``MeanCurvatureSolver``) together with the P1 level set and the isoparametric
deformation. This script extracts the **zero isoline on the deformed background
mesh** — i.e. the curved interface the solver integrates on — and colours it by
the magnitude of the mean curvature. Curvature is largest at the rounded top/
sides of the rising cap and small (or negative) along the flattened underside.

Usage::

    python viz_curvature_on_interface.py <probe_dir> [tag] [out.png]

or import :func:`plot_curvature`.
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import viz_common as V


def plot_curvature(probe_dir, tag="periodic", out="curvature_on_interface.png",
                   times=(0.5, 1.2, 1.9, 3.0), meta=None):
    """Draw the curved interface at several times, coloured by |H|, over the mesh."""
    pvd = os.path.join(probe_dir, f"{tag}_curv.pvd")
    tmap = V.pvd_times(pvd)
    if not tmap:  # single-file fallback
        tmap = {f: 0.0 for f in glob.glob(os.path.join(probe_dir, f"{tag}_curv*.vtu"))}
    metapath = meta or os.path.join(probe_dir, f"{tag}_meta.npz")
    tri = V.triangulation(metapath) if os.path.exists(metapath) else None

    # common colour scale across panels (robust to a few contour artefacts)
    allvals = []
    picks = []
    for tt in times:
        f = min(tmap, key=lambda k: abs(tmap[k] - tt))
        picks.append((f, tmap[f]))
        _, v = V.contour_segments(f, "H")
        if len(v):
            allvals.append(np.abs(v))
    vmax = np.percentile(np.concatenate(allvals), 98) if allvals else 4.0

    fig, axes = plt.subplots(1, len(picks), figsize=(3.0 * len(picks), 5.4), squeeze=False)
    lc = None
    for ax, (f, tt) in zip(axes[0], picks):
        if tri is not None:
            V.draw_mesh(ax, tri, color="#d5d9e0", lw=0.4)
        segs, vals = V.contour_segments(f, "H")
        lc = LineCollection(segs, cmap="viridis", linewidths=3.0)
        lc.set_array(np.abs(vals))
        lc.set_clim(0, vmax)
        ax.add_collection(lc)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 2)
        ax.set_title(f"t = {tt:.2f}", fontsize=10)
    if lc is not None:
        fig.colorbar(lc, ax=axes[0].tolist(), shrink=0.7, label="|mean curvature| $|H|$")
    fig.suptitle("Interface (curved, isoparametric) coloured by mean curvature", fontsize=12)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    probe_dir = sys.argv[1] if len(sys.argv) > 1 else "ht_probe"
    tag = sys.argv[2] if len(sys.argv) > 2 else "periodic"
    out = sys.argv[3] if len(sys.argv) > 3 else "curvature_on_interface.png"
    plot_curvature(probe_dir, tag, out)
