"""Before/after gallery of the redistancing steps.

Whenever the level set is redistanced, the ``ht_redist_probe.py`` driver dumps
the *continuous* level-set field just **before** and just **after** the
operation (``<tag>_redist<k>_before.vtu`` / ``_after.vtu``). This script draws,
for each such event, the level sets at $\\pm 0.025\\,i$ over the (deformed)
background mesh:

* the **zero** isoline (black, bold) is the interface — unchanged by
  redistancing;
* the inner (blue, $\\phi<0$) and outer (red, $\\phi>0$) isolines visualise
  $|\\nabla\\phi|$: **evenly spaced** means $|\\nabla\\phi|\\approx1$ (a clean
  signed-distance function), **bunched/spread** means the field has drifted.

This is the **higher-order level set drawn WITHOUT the isoparametric
deformation** — the representation that carries the signed-distance property and
is transported. (The deformation is only meaningful together with the $P^1$ cut,
which is the representation used for the geometry / numerical integration; the
two agree closely at the zero line by construction.) Do *not* warp this field by
``deform``.

Redistancing pulls the drifted *before* field back onto an evenly spaced *after*
field without moving the interface.

Usage::

    python viz_redistancing_gallery.py <probe_dir> [tag] [out.png]

or import :func:`plot_gallery`.
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pyvista as pv
import viz_common as V

STEP = 0.025   # level-set spacing; ALL multiples of STEP present in the domain are drawn


def _levels(vtu, step=STEP):
    """All isoline levels (multiples of ``step``) that actually occur in the
    domain, from the higher-order field's min to max."""
    phi = np.asarray(pv.read(vtu).point_data["phi"])
    lo = np.floor(float(phi.min()) / step) * step
    hi = np.ceil(float(phi.max()) / step) * step
    return list(np.round(np.arange(lo, hi + step / 2, step), 4))


def _color(level):
    if abs(level) < 1e-9:
        return "#111111", 2.6          # interface
    return ("#1f77b4" if level < 0 else "#d62728"), 0.8   # inside / outside


def _panel(ax, tri, vtu, title):
    V.draw_mesh(ax, tri, lw=0.5)
    # SDF view: the higher-order level-set field WITHOUT the deformation
    for lv, polys in V.isoline_levels(vtu, _levels(vtu), warp=False).items():
        col, lw = _color(lv)
        for p in polys:
            ax.plot(p[:, 0], p[:, 1], color=col, lw=lw)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)


def plot_gallery(probe_dir, tag="periodic", out="redistancing_gallery.png",
                 events=None, meta=None, pad=0.12):
    """Draw a row of before/after pairs, one per selected redistancing event.

    ``events`` selects which event indices to show (default: a few spread across
    the run). ``meta`` is the ``<tag>_meta.npz`` written by the probe (for the
    background mesh and the event times); if missing, the mesh is rebuilt at the
    stored ``maxh``.
    """
    befores = sorted(glob.glob(os.path.join(probe_dir, f"{tag}_redist*_before.vtu")))
    n = len(befores)
    if n == 0:
        raise FileNotFoundError(f"no {tag}_redist*_before.vtu in {probe_dir}")
    if events is None:
        events = sorted(set(np.linspace(0, n - 1, min(4, n)).round().astype(int)))

    metapath = meta or os.path.join(probe_dir, f"{tag}_meta.npz")
    times = None
    if os.path.exists(metapath):
        d = np.load(metapath)
        tri = V.triangulation(metapath)
        times = d["redist_times"] if "redist_times" in d else None
    else:  # rebuild the mesh from the maxh baked into a snapshot's sibling
        raise FileNotFoundError(f"{metapath} not found (needed for the background mesh)")

    # zoom to the union of all shown interfaces (undeformed, SDF view)
    allpts = np.vstack([np.vstack(V.interface_polylines(befores[k], warp=False) or [np.zeros((1, 2))])
                        for k in events])
    x0, x1 = allpts[:, 0].min() - pad, allpts[:, 0].max() + pad
    y0, y1 = allpts[:, 1].min() - pad, allpts[:, 1].max() + pad

    fig, axes = plt.subplots(2, len(events), figsize=(3.0 * len(events), 6.4), squeeze=False)
    for j, k in enumerate(events):
        tt = f" (t={times[k]:.2f})" if times is not None and k < len(times) else ""
        _panel(axes[0][j], tri, befores[k], f"event {k} — before{tt}")
        _panel(axes[1][j], tri, befores[k].replace("_before", "_after"), f"event {k} — after")
        for r in range(2):
            axes[r][j].set_xlim(x0, x1)
            axes[r][j].set_ylim(y0, y1)
    fig.suptitle(f"Redistancing before/after — level sets $\\pm0.025\\,i$ "
                 f"(black: interface, blue: $\\phi<0$, red: $\\phi>0$)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out} ({len(events)} events)")
    return out


if __name__ == "__main__":
    probe_dir = sys.argv[1] if len(sys.argv) > 1 else "ht_probe"
    tag = sys.argv[2] if len(sys.argv) > 2 else "periodic"
    out = sys.argv[3] if len(sys.argv) > 3 else "redistancing_gallery.png"
    plot_gallery(probe_dir, tag, out)
