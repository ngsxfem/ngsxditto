"""QoI histories vs. the Hysing–Turek reference.

Plots the three benchmark quantities of interest — rise velocity, circularity,
centroid height — as functions of time for one or more meshes, with the
reference values (and their inter-group range) marked. Reads the
``<label>_qoi.npz`` files written by ``ht_convergence.py``.

Usage::

    python viz_qoi_convergence.py <run_dir> [label1 label2 ...] [out.png]

or import :func:`plot_qoi_convergence`.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reference: (best single value, time) and the inter-group range (Hysing 2009, Table XII)
REF = {
    "vrise": dict(val=0.2417, t=0.92, lo=0.2417, hi=0.2421),
    "circ":  dict(val=0.9013, t=1.90, lo=0.9011, hi=0.9013),
    "yc":    dict(val=1.081,  t=3.00, lo=1.0799, hi=1.0817),
}
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]


def plot_qoi_convergence(run_dir, labels, out="qoi_convergence.png"):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for i, lab in enumerate(labels):
        d = np.load(os.path.join(run_dir, f"{lab}_qoi.npz"))
        c = COLORS[i % len(COLORS)]
        name = f"{lab} (ne={int(d['ne'])}, h={float(d['maxh']):.3f})"
        # clip a degenerating run (e.g. a too-coarse mesh) at the area blow-up so
        # the plot shows only the physical part of the curve
        a = d["area"]; ok = np.abs(a / a[0] - 1.0) < 0.1
        if not ok.all():
            ok[np.argmax(~ok):] = False
        t = d["t"][ok]
        ax[0].plot(t, d["vrise"][ok], color=c, lw=1.6, label=name)
        ax[1].plot(t, d["circ"][ok], color=c, lw=1.6, label=name)
        ax[2].plot(t, d["yc"][ok], color=c, lw=1.6, label=name)
    # reference bands + markers
    for a, key, title in [(ax[0], "vrise", "rise velocity $\\bar u_y$"),
                          (ax[1], "circ", "circularity $c$"),
                          (ax[2], "yc", "centroid height $y_c$")]:
        r = REF[key]
        a.axhspan(r["lo"], r["hi"], color="k", alpha=0.10)
        a.plot(r["t"], r["val"], "k*", ms=11)
        a.set_title(title); a.set_xlabel("t"); a.grid(alpha=.3)
    ax[2].legend(fontsize=8, loc="upper left")
    fig.suptitle("Hysing–Turek case 1 — QoI vs. reference "
                 "(★ best value, grey band = inter-group range)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    run_dir = args.pop(0) if args else "ht_runs"
    out = "qoi_convergence.png"
    if args and args[-1].endswith(".png"):
        out = args.pop()
    labels = args or ["coarse", "medium", "fine"]
    plot_qoi_convergence(run_dir, labels, out)
