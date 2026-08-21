# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Redistancing as a distance solver: distance to a door
#
# Before the abstract two-phase level-set setting (see
# `redistancing_initializers.ipynb`), here is the most tangible possible picture
# of *what redistancing computes*: the distance, within a room, to the door.
#
# Redistancing turns an arbitrary field into a signed-distance function by
# iterating `|grad(phi)| = 1` while keeping `phi = 0` fixed on a source set.
# `MinimizationBasedRedistancing` does exactly this on an UNFITTED level set:
# it pins `phi = 0` on the cut interface (via `dCut(...)`) and, each
# iteration, solves the lagged-diffusivity linear system
#
# ```
# grad(phi_new) . grad(v) dX  +  alpha * phi_new * v * dS_source
#     =  (1/|grad(phi_cur)|) * grad(phi_cur) . grad(v) dX .
# ```
#
# This script runs the *identical* iteration, but on a plain FITTED mesh with
# the source being a real mesh BOUNDARY segment (the "door", via `ds(...)`)
# rather than a level-set interface -- so `phi` becomes the (unsigned)
# distance to the door. No unfitted/CutFEM machinery at all; the room even has a
# few tables cut out as holes, so the mesh geometry is non-trivial.
#
# It also shows an important limitation that directly motivates the
# `initializer=` option in `redistancing_initializers.ipynb`: this elliptic
# relaxation is diffusive, so it produces a smooth distance-*like* field but does
# NOT resolve the causal "shadow" behind an obstacle (a point tucked behind a
# table comes out at almost the same value as an equally-far point in the open).
# A causal method -- graph shortest-path / fast marching -- does resolve it. In
# the level-set setting that causal method is exactly `FastMarching`, which is
# why using it as an `initializer=` for the elliptic minimizer gives the best
# of both worlds: the causal method lays down the correct global structure, the
# elliptic minimizer polishes it to `|grad(phi)| = 1`.

# %%
import heapq
import numpy as np
from netgen.occ import WorkPlane, OCCGeometry
from ngsolve import *
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# %% [markdown]
# ## Fitted room geometry: door boundary segment + tables as holes

# %%
W, H = 8.0, 5.0
door_lo, door_hi = 3.0, 4.0

wp = WorkPlane().MoveTo(0, 0)
wp.LineTo(door_lo, 0, "wall")
wp.LineTo(door_hi, 0, "door")          # <- the named source boundary segment
wp.LineTo(W, 0, "wall")
wp.LineTo(W, H, "wall")
wp.LineTo(0, H, "wall")
wp.LineTo(0, 0, "wall")
room = wp.Face()

tables_xywh = [(1.2, 1.0, 1.4, 0.9), (1.2, 3.2, 1.4, 0.9), (4.0, 2.0, 1.6, 1.0),
               (6.2, 0.8, 1.2, 1.2), (6.2, 3.3, 1.2, 1.0)]
domain = room
for (x0, y0, w, h) in tables_xywh:
    domain = domain - WorkPlane().MoveTo(x0, y0).Rectangle(w, h).Face()
for e in domain.edges:
    if e.name is None:
        e.name = "table"

mesh = Mesh(OCCGeometry(domain, dim=2).GenerateMesh(maxh=0.12))
print(f"room mesh: {mesh.ne} elements, {mesh.nv} vertices")

# %% [markdown]
# ## (1) Elliptic distance-to-door: the MinimizationBasedRedistancing iteration

# %%
order, n_iter = 2, 25
fes = H1(mesh, order=order)
phi, v = fes.TnT()
dX = dx()
dS_door = ds(definedon=mesh.Boundaries("door"))    # pin phi=0 on the door

a = BilinearForm(fes, check_unused=False)
a += grad(phi) * grad(v) * dX
a += 1e4 * phi * v * dS_door
a.Assemble()
inv = a.mat.Inverse(fes.FreeDofs())

phi_ell = GridFunction(fes)
phi_ell.Set(sqrt((x - 0.5 * (door_lo + door_hi)) ** 2 + y ** 2))   # any nonzero-gradient start
for _ in range(n_iter):
    b = LinearForm(fes)
    b += (1.0 / Norm(grad(phi_ell))) * grad(phi_ell) * grad(v) * dX
    b.Assemble()
    phi_ell.vec.data = inv * b.vec
res = sqrt(Integrate((Norm(grad(phi_ell)) - 1) ** 2, mesh) / Integrate(CF(1), mesh))
print(f"elliptic relaxation: rms(|grad phi| - 1) = {res:.4f}")

# %% [markdown]
# ## (2) Causal distance-to-door: graph shortest path (Dijkstra) from the door
#
# Same idea as `FastMarching`, propagating along mesh edges -- obstacles (holes)
# have no edges through them, so the distance must genuinely detour around them.

# %%
V1 = H1(mesh, order=1)
v2d = {vtx: V1.GetDofNrs(vtx)[0] for vtx in mesh.vertices}
d2v = {d: vtx for vtx, d in v2d.items()}
dist = {d: float("inf") for d in v2d.values()}
door_vs = {vtx for el in mesh.Elements(BND) if el.mat == "door" for vtx in el.vertices}
pq = []
for vtx in door_vs:
    dist[v2d[vtx]] = 0.0
    heapq.heappush(pq, (0.0, v2d[vtx]))
done = set()
while pq:
    d_cur, cur = heapq.heappop(pq)
    if cur in done:
        continue
    done.add(cur)
    p_cur = mesh[d2v[cur]].point
    for edge in mesh[d2v[cur]].edges:
        a_, b_ = mesh[edge].vertices
        nxt = v2d[b_ if a_ == d2v[cur] else a_]
        if nxt in done:
            continue
        nd = d_cur + np.hypot(*(np.array(p_cur) - np.array(mesh[d2v[nxt]].point)))
        if nd < dist[nxt]:
            dist[nxt] = nd
            heapq.heappush(pq, (nd, nxt))
phi_causal = GridFunction(V1)
for vtx, d in v2d.items():
    phi_causal.vec[d] = dist[d]

# %% [markdown]
# ## Figure

# %%
verts = np.array([list(p.point) for p in mesh.vertices])
tris = np.array([[w.nr for w in el.vertices] for el in mesh.Elements(VOL)])
tri = mtri.Triangulation(verts[:, 0], verts[:, 1], tris)
vals_ell = np.array([phi_ell(mesh(px, py)) for px, py in verts])
vals_cau = np.array([phi_causal.vec[v2d[vtx]] for vtx in mesh.vertices])

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
vmax = max(vals_ell.max(), vals_cau.max())
levels = np.linspace(0, vmax, 24)
for ax, vals, title in [
        (axes[0], vals_ell, "elliptic relaxation (= redistancer iteration)\nsmooth, but smears through tables"),
        (axes[1], vals_cau, "causal graph distance (= FastMarching idea)\ndetours correctly around tables")]:
    cf = ax.tricontourf(tri, vals, levels=levels, cmap="viridis")
    ax.tricontour(tri, vals, levels=levels, colors="white", linewidths=0.4)
    ax.plot([door_lo, door_hi], [0, 0], color="red", lw=4, solid_capstyle="butt", label="door")
    for (x0, y0, w, h) in tables_xywh:
        ax.add_patch(plt.Rectangle((x0, y0), w, h, facecolor="lightgray", edgecolor="k"))
    ax.set_aspect("equal"); ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11)
axes[0].legend(loc="upper right", fontsize=8)
fig.colorbar(cf, ax=axes, label="distance to the door", shrink=0.85)
fig.suptitle('Redistancing as a distance solver: "distance to the door" on a fitted room mesh', fontsize=13)
fig.savefig("redistancing_distance_to_door.png", dpi=140, bbox_inches="tight")
plt.show()
