"""Pluggable initializers for MinimizationBasedRedistancing.

MinimizationBasedRedistancing only ever *corrects* whatever field it is
handed: each Newton/fixed-point iteration nudges phi towards |grad(phi)|=1
while pinning it to zero on the (P1-interpolated) interface. Handed a rough
or noisy input -- e.g. a level set that has accumulated high-frequency
dispersion noise from many explicit transport steps -- it needs many
iterations to smooth that out, and far from the interface (where the pinning
term has no direct influence) that convergence is particularly slow.

`initializer=` lets another redistancer clean up/replace the input ONCE,
in place, before the main iteration starts:

* `FastMarching()` -- a causally correct global distance computation
  (graph propagation from the interface). Gets the right answer near
  singular/medial-axis-like structure essentially for free, at the cost of
  its own mesh-graph discretization noise (which the main iteration then
  polishes away).
* `GPExtensionRedistancing()` -- NOT a real redistancer (does not aim for
  |grad(phi)|=1 at all), just one cheap ghost-penalty diffusion solve pinned
  to the interface. Directly targets removing exactly the kind of
  high-frequency noise a drifted level set tends to carry.

This script builds a synthetic "drifted-like" input (true signed distance to
a circle, plus added high-frequency noise, mimicking transport dispersion)
and compares: no initializer, FastMarching, GPExtensionRedistancing.
"""
import numpy as np
from ngsolve import *
from netgen.geom2d import SplineGeometry
from ngsxditto.redistancing import (MinimizationBasedRedistancing, FastMarching,
                                     GPExtensionRedistancing)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

domain = SplineGeometry()
domain.AddCircle((0, 0), 1)
mesh = Mesh(domain.GenerateMesh(maxh=0.08))

order = 2
n_iter = 10
true_sdf = sqrt(x ** 2 + y ** 2) - 0.5
# synthetic stand-in for transport-dispersion noise on a drifted level set
rough = true_sdf + 0.08 * sin(15 * x) * cos(13 * y) + 0.04 * sin(29 * x) * sin(23 * y)


def l2_err(phi):
    return Integrate((phi - true_sdf) ** 2, mesh) ** 0.5 / Integrate(CF(1), mesh)


def grad_rms(phi):
    return (Integrate((Norm(grad(phi)) - 1) ** 2, mesh) / Integrate(CF(1), mesh)) ** 0.5


variants = [("baseline\n(no initializer)", None),
            ("+ FastMarching\ninitializer", FastMarching()),
            ("+ GPExtensionRedistancing\ninitializer", GPExtensionRedistancing())]

print(f"{'variant':32s} {'L2 err vs true SDF':>20s} {'rms(|grad phi|-1)':>20s}")
results = []
fields = []
for name, init in variants:
    phi = GridFunction(H1(mesh, order=order))
    phi.Set(rough)
    MinimizationBasedRedistancing(alpha=10000, n_iter=n_iter, initializer=init).Redistance(phi)
    e, g = l2_err(phi), grad_rms(phi)
    results.append((name, e, g))
    fields.append(phi)
    print(f"{name.replace(chr(10), ' '):32s} {e:20.5f} {g:20.5f}")

# for reference: GPExtensionRedistancing alone, to make explicit that it does
# NOT solve the Eikonal equation on its own (that's the whole point of it
# only ever being used as an initializer, never as the final redistancer)
phi_gp_alone = GridFunction(H1(mesh, order=order)); phi_gp_alone.Set(rough)
GPExtensionRedistancing().Redistance(phi_gp_alone)
print(f"\n{'GPExtensionRedistancing ALONE (not a real redistancer)':32s} "
      f"{l2_err(phi_gp_alone):20.5f} {grad_rms(phi_gp_alone):20.5f}  <- rms(|grad phi|-1) stays large, as expected")

# -------------------------------------------------------------------- plot --
fig, axes = plt.subplots(1, len(variants), figsize=(5 * len(variants), 5))
levels = np.linspace(-0.5, 0.5, 21)
for ax, (name, _), phi in zip(axes, variants, fields):
    xs = ys = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = np.full_like(X, np.nan)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            px, py = X[i, j], Y[i, j]
            if px ** 2 + py ** 2 < 0.98:
                Z[i, j] = phi(mesh(px, py))
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="RdBu_r")
    ax.contour(X, Y, Z, levels=[0.0], colors="black", linewidths=2)
    ax.set_aspect("equal")
    ax.set_title(name)
fig.colorbar(cf, ax=axes, shrink=0.8, label="phi")
fig.suptitle("MinimizationBasedRedistancing: effect of the initializer= choice\n"
             "(synthetic noisy input, same n_iter for all three)")
fig.savefig("redistancing_initializers.png", dpi=140, bbox_inches="tight")
print("\nwrote redistancing_initializers.png")
