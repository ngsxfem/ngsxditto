"""Visualise the velocity field around the rising bubble.

The ``ht_velocity_dump.py`` driver writes, at several times, the two-phase
velocity ``vel`` = ``IfPos(lset, u_pos, u_neg)`` together with the P1 level set,
the isoparametric deformation and (in ``vel_meta.npz``) the mean bubble rise
velocity $\\bar u_y(t)$. This script shows the flow

* in the **laboratory frame** (absolute velocity), and
* in the **bubble frame**, i.e. relative to the mean bubble velocity
  $\\mathbf u-(0,\\bar u_y)$, which exposes the recirculation around the bubble,

as streamlines over a colour map of the speed, with the interface drawn on top.

Usage::

    python viz_velocity_field.py <vel_dir> [out_prefix]

or import :func:`plot_velocity`.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyvista as pv
pv.OFF_SCREEN = True

import viz_common as V


def _sample(vtu, nx=90, ny=180):
    """Sample the (deformed) velocity field onto a uniform grid over [0,1]x[0,2].
    Returns X, Y (grid), U, V (masked velocity components)."""
    m = V.read_warped(vtu, warp=True)          # geometry = P1 cut + deformation
    grid = pv.ImageData(dimensions=(nx, ny, 1),
                        spacing=(1.0 / (nx - 1), 2.0 / (ny - 1), 1.0), origin=(0, 0, 0))
    s = grid.sample(m)
    vel = np.asarray(s["vel"]).reshape(ny, nx, -1)
    valid = np.asarray(s["vtkValidPointMask"]).reshape(ny, nx).astype(bool)
    U = np.where(valid, vel[:, :, 0], np.nan)
    Vc = np.where(valid, vel[:, :, 1], np.nan)
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 2, ny))
    return X, Y, U, Vc


def plot_velocity(vel_dir, out_prefix="velocity", times=(0.5, 1.0, 1.9, 3.0)):
    meta = np.load(os.path.join(vel_dir, "vel_meta.npz"))
    tmap = V.pvd_times(os.path.join(vel_dir, "vel.pvd"))
    tri = V.triangulation((meta["verts"], meta["tris"]))
    picks = []
    for tt in times:
        if not tmap:
            continue
        f = min(tmap, key=lambda k: abs(tmap[k] - tt))
        ts = tmap[f]
        vr = float(np.interp(ts, meta["t"], meta["vrise"]))
        picks.append((f, ts, vr))

    # sample once per pick (frame-independent); reuse for both frames
    sampled = [(f, ts, vr) + _sample(f) for f, ts, vr in picks]   # (f,ts,vr,X,Y,U,Vc)

    for frame in ("lab", "bubble"):
        fig, axes = plt.subplots(1, len(picks), figsize=(3.0 * len(picks), 5.6), squeeze=False)
        vfields = [(Vc - (vr if frame == "bubble" else 0.0)) for (f, ts, vr, X, Y, U, Vc) in sampled]
        smax = max(np.nanpercentile(np.hypot(U, Vf), 98)
                   for (f, ts, vr, X, Y, U, Vc), Vf in zip(sampled, vfields)) or 1.0
        for ax, (f, ts, vr, X, Y, U, Vc), Vf in zip(axes[0], sampled, vfields):
            ax.pcolormesh(X, Y, np.hypot(U, Vf), cmap="viridis", vmin=0, vmax=smax,
                          shading="auto", rasterized=True)
            with np.errstate(invalid="ignore"):
                ax.streamplot(X, Y, np.nan_to_num(U), np.nan_to_num(Vf), color="white",
                              density=1.1, linewidth=0.6, arrowsize=0.6)
            for p in V.interface_polylines(f, warp=True):
                ax.plot(p[:, 0], p[:, 1], color="#d62728", lw=1.8)
            ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 2)
            ax.set_title(f"t = {ts:.2f}", fontsize=10)
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, smax))
        cb = fig.colorbar(sm, ax=axes[0].tolist(), shrink=0.7)
        cb.set_label("speed" + (" (bubble frame)" if frame == "bubble" else ""))
        title = ("Velocity relative to the mean bubble rise velocity (bubble frame)"
                 if frame == "bubble" else "Velocity field (laboratory frame)")
        fig.suptitle(title, fontsize=12)
        out = f"{out_prefix}_{'relative' if frame == 'bubble' else 'absolute'}.png"
        fig.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    vel_dir = sys.argv[1] if len(sys.argv) > 1 else "ht_vel"
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "velocity"
    plot_velocity(vel_dir, out_prefix)
