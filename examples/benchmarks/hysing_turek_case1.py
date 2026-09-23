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
# # Rising bubble benchmark (Hysing–Turek, test case 1)
#
# > **This is a benchmark example** — a heavier, quantitative validation study
# > that is *not executed* by the documentation build (see the *Benchmark
# > examples* section). The code below is fully runnable, but the reported
# > numbers and all figures are **pre-computed and embedded**; reproduce them by
# > running this notebook yourself — the per-mesh run times are listed in the
# > comparison table below (a few minutes on a coarse mesh, hours on the finest).
#
# We reproduce the classical 2D **rising-bubble benchmark** of Hysing, Turek et
# al. [1] with `ngsxditto`'s unfitted two-phase Navier–Stokes solver and check
# the computed *quantities of interest* (QoI) against the published reference
# values.
#
# **Setup.** A circular gas bubble rises under buoyancy in a heavier liquid
# column $\Omega = [0,1]\times[0,2]$; the bubble has radius $r_0=0.25$ and is
# initially centred at $(0.5,0.5)$. Gravity is $\mathbf g=(0,-0.98)$; the walls
# are **no-slip** on top/bottom and **free-slip** (no penetration) on left/right.
# We integrate over $t\in[0,3]$.
#
# | phase | density $\rho$ | dyn. viscosity $\mu$ | surface tension $\sigma$ |
# |-------|:-----:|:-----:|:-----:|
# | bubble (inside, `fluid1`, levelset $<0$) | $100$  | $1$  | $24.5$ |
# | liquid (outside, `fluid2`, levelset $>0$) | $1000$ | $10$ | $24.5$ |
#
# This gives Eötvös number $\mathrm{Eo}=\rho_2\,g\,(2r_0)^2/\sigma=10$ and
# Reynolds number $\mathrm{Re}=\rho_2\sqrt{g\,(2r_0)^3}/\mu_2=35$ — an
# **inertial** regime, so the nonlinear convection term is essential.
#
# **Reference QoI.** The benchmark was computed independently by three groups
# (TP2D, FreeLIFE, MooNMD); the spread between their finest-grid results is a
# rough *confidence interval* — the accuracy floor any computation can be checked
# against (Hysing et al. 2009, Table XII):
#
# | quantity | TP2D | FreeLIFE | MooNMD | reference range |
# |----------|:---:|:---:|:---:|:---:|
# | max. rise velocity (@ $t\approx0.92$) | 0.2417 | 0.2421 | 0.2417 | **0.2417–0.2421** |
# | min. circularity (@ $t\approx1.9$) | 0.9013 | 0.9011 | 0.9013 | **0.9011–0.9013** |
# | centroid $y_c(3)$ | 1.0813 | 1.0799 | 1.0817 | **1.0799–1.0817** |
#
# The spread between the three groups is the benchmark's own uncertainty; a
# computation landing inside the **reference range** (bold) is as good as the
# benchmark can resolve.
#
# > [1] S. Hysing, S. Turek, D. Kuzmin, N. Parolini, E. Burman, S. Ganesan,
# > L. Tobiska, *Quantitative benchmark computations of two-dimensional bubble
# > dynamics*, Int. J. Numer. Meth. Fluids **60** (2009) 1259–1288.

# %%
import math
import os
import numpy as np

from ngsxditto import *
from ngsolve import *
from xfem import *
from netgen.occ import OCCGeometry, MoveTo, X, Y

# %% [markdown]
# ## The discretisation, assembled once as `run_case`
#
# The full simulation combines the `ngsxditto` building blocks:
#
# * **`TwoPhaseTaylorHood`** — unfitted two-phase Navier–Stokes with the
#   symmetric-gradient viscous stress $2\mu\,\varepsilon(\mathbf u)$ (mandatory
#   across a viscosity jump), a Nitsche coupling of the two phases, ghost-penalty
#   stabilisation and a curvature surface-tension force. We enable the nonlinear
#   convection (`add_convection=True`) and second-order BDF2 stepping
#   (`time_order=2`). The **velocity space is `order+1`** (Taylor–Hood), while
#   the geometry/levelset order is `order` — so `order=2` already means a
#   quadratic isoparametric interface and a $P^3$ velocity. This extra velocity
#   order matters here: dropping to an equal $P^2/P^1$ velocity (velocity order 2)
#   overshoots the rise-velocity peak markedly ($v_{\max}\approx0.2576$, $+6.6\%$
#   vs. the reference $0.2417$–$0.2421$) and adds oscillation, so we keep
#   `order+1`.
# * **`LevelSetGeometry`** with `ExplicitDGTransport` and the high-order
#   variational **`MinimizationBasedRedistancing`** (kept sharp every 20 steps by
#   `PeriodicRedistancing`).
# * **`MeanCurvatureSolver`** for the interface curvature that drives the surface
#   tension — wired into the level set's own update callback
#   (`levelset.AddCallback(mean_curvature.Step, index=1)`) rather than registered
#   as its own time-loop step, so it always sees the just-transported geometry
#   (see *Second order in time*).
# * **`LevelsetBasedExtension`** to extend the interface velocity — run *after*
#   the fluid solve, so it extends the just-computed velocity, not the previous
#   step's — feeding an `Extrapolator` (via `FeedInto`/`Predictor`) as the
#   **interval-centred transport wind**.
# * Two flags on `TwoPhaseTaylorHood` control how the convective nonlinearity is
#   linearized: `linearization` (`"newton"` or `"picard"`, i.e. the classical
#   Oseen fixed-point) and `extrapolated_advection` (`True`/`False`) — see
#   *Second order in time* for why both matter and how they combine to give
#   second order **without** any Picard sub-iterations.
#
# > **⚠️ Body-force convention.** The body force passed as `f1`/`f2` is the
# > **acceleration** $\mathbf g$, *not* the force density $\rho\,\mathbf g$:
# > `TwoPhaseTaylorHood` multiplies it by the phase density internally (the
# > momentum body force is assembled as $\rho_i\,f_i$).

# %%
def run_case(maxh, dt, order=2, end_time=3.0, n_subiter=1, wind="extrapolate", redist=True,
            linearization="picard", extrapolated_advection=True,
            snapshot_dir=None, snapshot_label="snap", n_snapshots=40,
            snapshot_subdivision=4):
    """Run Hysing–Turek case 1 on one mesh/time-step and return the QoI history.

    `wind` selects the transport wind: `"endpoint"` (the value at t^{n+1}, 1st
    order in time) or `"extrapolate"` (interval-centred via an Extrapolator, 2nd
    order). `redist` toggles the redistancing; turn it off for a clean
    dt-refinement study (its step-based schedule is otherwise dt-dependent).
    `linearization`/`extrapolated_advection` control the convective
    linearization -- see the "Second order in time" section for why the default
    (`"picard"`, `True`) already reaches second order at `n_subiter=1`, with no
    Picard iteration at all.

    Passing `snapshot_dir` writes `<snapshot_label>_snap*.vtu` (plus a
    `_meta.npz` with the background mesh) for the visualisation scripts.

    `snapshot_subdivision` controls how finely the level set is sampled when it
    is written out, and it matters more than it looks: the level set is P1 and
    the geometry lives in the *isoparametric deformation*, so an interface
    written with too little subdivision is drawn as one straight chord per cut
    element. On a coarse mesh that reads as a kinked, oscillating interface even
    when the computed interface is smooth -- at h=0.08 a perfectly smooth
    analytic ellipse picks up an apparent high-frequency deviation of 7.3e-3
    with `subdivision=0` versus 1.1e-3 with `subdivision=4`. Keep this at 3-4
    for anything that ends up in a figure."""
    g = 0.98
    rho1, mu1, sigma = 100.0, 1.0, 24.5   # bubble  (inside, negative levelset)
    rho2, mu2        = 1000.0, 10.0       # liquid  (outside, positive levelset)

    # --- mesh: rectangle with named walls -----------------------------------
    domain = MoveTo(0, 0).Rectangle(1, 2).Face()
    domain.edges.Min(X).name = "left";   domain.edges.Max(X).name = "right"
    domain.edges.Min(Y).name = "bottom"; domain.edges.Max(Y).name = "top"
    mesh = Mesh(OCCGeometry(domain, dim=2).GenerateMesh(maxh=maxh))

    # --- levelset: transport + high-order variational redistancing ----------
    t = Parameter(0.0)
    transport = ExplicitDGTransport(mesh, dt=dt, order=order, compile=False)
    if redist:
        levelset = LevelSetGeometry(transport, redistancing=MinimizationBasedRedistancing(),
                                    autoredistancing=PeriodicRedistancing(20))
    else:
        levelset = LevelSetGeometry(transport)          # clean dt study (see below)
    levelset.Initialize(sqrt((x - 0.5)**2 + (y - 0.5)**2) - 0.25)

    # --- two-phase Navier–Stokes --------------------------------------------
    # surface tension is one interface property -- the solver reads only fluid1's
    # coefficient, so fluid2 does not carry one
    fluid1_params = FluidParameters(viscosity=mu1, density=rho1)
    fluid2_params = FluidParameters(viscosity=mu2, density=rho2)
    mean_curvature = MeanCurvatureSolver(mesh, order=order, lset=levelset, gp_param=1)
    mean_curvature.Step()          # prime H before fluid is constructed
    grav = CF((0, -g))
    fluid = TwoPhaseTaylorHood(
        mesh, fluid1_params=fluid1_params, fluid2_params=fluid2_params, lset=levelset, surface_tension_coeff=sigma,
        surface_tension=mean_curvature.H, f1=grav, f2=grav,   # acceleration, see warning
        dt=dt, order=order + 1, time_order=2, advection=True,
        ghost_stab=1, nitsche_stab=100,
        linearization=linearization, extrapolated_advection=extrapolated_advection)
    fluid.SetOuterBoundaryCondition(StrongDirichletBC(region="top|bottom", values=CF((0, 0))))
    fluid.SetOuterBoundaryCondition(NitscheNormalVelocityBC(region="left|right", values=CF(0)))
    fluid.Initialize()

    # curvature rides along inside the level set's own update callback -- always
    # computed on the just-transported (t^{n+1}) geometry, never a step behind
    # (see "Second order in time")
    levelset.AddCallback(mean_curvature.Step, index=1)

    # --- coupling: interface velocity -> transport wind ---------------------
    # The extension runs AFTER the fluid solve (below), so it extends the
    # just-computed velocity. Feeding it into an Extrapolator and evaluating at
    # the interval midpoint t-dt/2 gives the 2nd-order interval-centred wind.
    velocity_extension = LevelsetBasedExtension(levelset, gamma=1e-3, order=order)
    velocity_extension.SetRhs(fluid.gfu.components[0])
    velocity_extension.Step()      # initial value at t=0
    if wind == "extrapolate":
        wind_predictor = velocity_extension.Predictor(time=t, order=1, offset=-dt / 2)
        levelset.transport.SetWind(wind_predictor.gf)
    else:
        levelset.transport.SetWind(velocity_extension.field)

    # --- QoI over the bubble region Omega_1 (negative levelset) -------------
    #   centroid height  y_c = (∫ y) / (∫ 1)
    #   rise velocity    v   = (∫ u_y) / (∫ 1)
    #   circularity      c   = 2 sqrt(pi |Ω1|) / |∂Ω1|   (= 1 for a circle)
    # --- optional snapshots for the visualisation scripts --------------------
    # NOTE the field is written as "phi": that is the name viz_common expects.
    _snap_vtk = None
    if snapshot_dir is not None:
        os.makedirs(snapshot_dir, exist_ok=True)
        _snap_vtk = VTKOutput(ma=mesh,
                              coefs=[levelset.lsetadap.lset_p1, levelset.lsetadap.deform],
                              names=["phi", "deform"],
                              filename=os.path.join(snapshot_dir, f"{snapshot_label}_snap"),
                              subdivision=snapshot_subdivision, floatsize="single")
        _snap_every = max(1, int(round(end_time / dt / max(1, n_snapshots))))
        _snap_count = [0]

    u_neg_y = fluid.gfu.components[0][1]
    hist = {"t": [], "yc": [], "vrise": [], "circ": [], "area": []}
    def record_qoi():
        area = Integrate(CF(1) * levelset.dx_neg, mesh)
        if area <= 1e-12:
            return
        perim = Integrate(CF(1) * levelset.dS, mesh)
        hist["t"].append(t.Get())
        hist["yc"].append(Integrate(y * levelset.dx_neg, mesh) / area)
        hist["vrise"].append(Integrate(u_neg_y * levelset.dx_neg, mesh) / area)
        hist["circ"].append(2.0 * math.sqrt(math.pi * area) / perim if perim > 0 else 0.0)
        hist["area"].append(area)
        if _snap_vtk is not None:
            _snap_count[0] += 1
            if _snap_count[0] % _snap_every == 0:
                _snap_vtk.Do(time=t.Get())

    # --- time loop ------------------------------------------------------------
    # module order: level set moves first (curvature rides along via its
    # callback) -> fluid solves on the fresh geometry -> extension uses the
    # just-solved velocity. With n_subiter=1 (the default) this already reaches
    # 2nd order in time, no Picard sub-iteration needed at all.
    time_loop = TimeLoop(time=t, dt=dt, end_time=end_time, display_progress_bar=True)
    if n_subiter > 1:
        time_loop.SetFinalizeRule(lambda: time_loop.i_inner >= n_subiter)
    if wind == "extrapolate":
        time_loop.Register(wind_predictor,     name="wind eval")
    time_loop.Register(levelset,           name="levelset")
    time_loop.Register(fluid,              name="two-phase flow")
    time_loop.Register(velocity_extension, name="vel ext.")
    time_loop.Register(FunctionCallStepper(record_qoi, as_validate=True), name="qoi")

    record_qoi()          # initial state, t = 0
    try:
        time_loop()
    except Exception as exc:
        # the P1 cut can degenerate at the strongest deformation ("Cutting this
        # part of a tetraeder ..."); keep the history collected up to that point
        print(f"  stopped early at t={t.Get():.3f}: {exc}")
    if snapshot_dir is not None:
        np.savez(os.path.join(snapshot_dir, f"{snapshot_label}_qoi.npz"),
                 verts=np.array([list(v.point) for v in mesh.vertices]),
                 tris=np.array([[w.nr for w in el.vertices] for el in mesh.Elements(VOL)]),
                 ne=mesh.ne, order=order,
                 **{k: np.array(v) for k, v in hist.items()})
    return {k: np.array(v) for k, v in hist.items()}, mesh.ne


# %% [markdown]
# ## Convergence to the reference values
#
# We run three successively finer meshes (halving both $h$ and $\Delta t$ each
# level) with `run_case`'s defaults — interval-centred wind, `linearization=
# "picard"`, `extrapolated_advection=True`, **`n_subiter=1`** (no Picard
# iteration at all — see *Second order in time* for why this already suffices) —
# and compare the QoI against the published reference values (and their
# inter-group range). All runs use `order=2` (quadratic isoparametric geometry,
# $P^3$ velocity).

# %%
cases = {
    #  level  maxh    dt       order    (h and dt both halved level to level)
    "L0":    dict(maxh=0.08, dt=0.010,  order=2),
    "L1":    dict(maxh=0.04, dt=0.005,  order=2),
    "L2":    dict(maxh=0.02, dt=0.0025, order=2),
}
results = {}
for label, cfg in cases.items():
    results[label] = run_case(**cfg)      # n_subiter=1, picard + extrapolated_advection (defaults)

# %% [markdown]
# ### Results
#
# The three levels, their QoI and their wall-clock run times (all on 44 cores):
#
# | level | $h$ | $\Delta t$ | elements | $v_{\max}$ | $c_{\min}$ | $y_c(3)$ | wall time |
# |---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
# | L0 | 0.08 | 0.010 | 718 | 0.2444 | 0.9003 | 1.0806 | 5.3 min |
# | L1 | 0.04 | 0.005 | 2900 | 0.2425 | 0.9017 | 1.0816 | 42 min |
# | L2 | 0.02 | 0.0025 | 11600 | 0.2419 | 0.9014 | 1.0817 | 2.9 h |
# | **reference** ★ | — | — | — | **0.2417–0.2421** | **0.9011–0.9013** | **1.0799–1.0817** | — |
#
# The maximum rise velocity converges **monotonically into the reference range**
# ($0.2444\to0.2425\to\mathbf{0.2419}\in[0.2417,0.2421]$); the centroid heights all
# lie inside $[1.0799,1.0817]$; the circularity minimum brackets and approaches
# $[0.9011,0.9013]$. Even the finest mesh here — 11 600 elements, still coarse for
# this benchmark — matches the reference to within its own inter-group uncertainty.
#
# These run times are with **no Picard sub-iteration** — about 3–7$\times$ faster
# than an earlier, otherwise identical campaign that needed `n_subiter=3` to reach
# the same accuracy (see *Second order in time*). (Run times still grow with the
# linear-solve cost, which dominates; see *Where the time goes* below.)

# %% [markdown]
# ### QoI histories vs. the reference
#
# The rise velocity climbs to its maximum near $t\approx0.9$, the circularity dips
# as the bubble forms an ellipsoidal cap (minimum near $t\approx1.9$), and the
# centroid rises steadily. Both meshes track each other and the reference (★, grey
# band = inter-group range) closely over the full $t\in[0,3]$; the coarse mesh (L0)
# shows a small circularity **wobble in the tail** ($t\gtrsim2$) that the finer mesh
# removes — a cut-cell resolution effect (the solver carries no convection
# stabilisation, and adding ghost penalty only makes it worse), not a coupling
# artefact.
#
# ![QoI histories for the two clean-to-$t=3$ meshes, with the Hysing–Turek reference values marked.](hysing_turek/qoi_convergence.png)

# %% [markdown]
# ## Second order in time
#
# `run_case` feeds the transport wind through an `Extrapolator` evaluated at the
# **interval midpoint** $t-\tfrac{\Delta t}{2}$ rather than using the end-point
# value at $t^{n+1}$. This alone is not sufficient: with the end-point wind, or
# with only a single Picard/Newton pass per step (`n_subiter=1`) and the *old*
# convective linearization (around the current, never-updated iterate), the
# centroid stays first order in time. Reaching second order used to need several
# (`n_subiter=3`) converged sub-iterations per step — three to four times the
# cost of a single solve.
#
# Getting to second order **without** any sub-iteration turned out to need two
# more fixes, beyond the interval-centred wind, for two independent first-order
# error sources that sub-iterations happened to mask:
#
# 1. **Curvature timing.** `MeanCurvatureSolver` reads the level set's *shared*
#    geometry; registered as its own time-loop step *before* the level set moves,
#    it always computed $H$ on the *previous* step's geometry (self-healing only
#    across several Picard passes). Fix: wire it into the level set's own update
#    callback (`levelset.AddCallback(mean_curvature.Step, index=1)`), so it always
#    runs on the just-transported ($t^{n+1}$) geometry.
# 2. **Wind-feed staleness.** The velocity extension used to run *before* the
#    fluid solve, so the sample it fed into the wind history was an extension of
#    the *old* $u^n$, not the just-solved $u^{n+1}$ — a full-step lag in the wind
#    history, independent of the interval-centred midpoint evaluation itself. Fix:
#    run the extension *after* the fluid solve.
#
# With both fixes, `TwoPhaseTaylorHood`'s convective term can be linearized around
# an **extrapolated, sub-iteration-refined predictor** for $u^{n+1}$
# (`extrapolated_advection=True`) instead of the current Picard/Newton iterate —
# orthogonal to `linearization` (`"newton"`'s full Jacobian + residual correction,
# or `"picard"`'s plain Oseen fixed-point): both reach second order once $\beta$ is
# the extrapolated predictor. A clean $\Delta t$ study at a fixed mesh
# (redistancing off, so its step-based schedule does not pollute the
# self-convergence) confirms it on the centroid — the QoI most sensitive to the
# temporal error, as it accumulates $\int u_y\,\mathrm dt$ over the trajectory:

# %%
def centroid_temporal_order(linearization, extrapolated_advection, n_subiter=1, dts=(0.01, 0.005, 0.0025)):
    hs = [run_case(maxh=0.05, dt=dt, end_time=1.0, n_subiter=n_subiter, redist=False,
                   linearization=linearization, extrapolated_advection=extrapolated_advection)[0]
          for dt in dts]
    tg = np.linspace(0.0, 1.0, 401)
    yc = [np.interp(tg, h["t"], h["yc"]) for h in hs]
    d01 = np.sqrt(np.mean((yc[0] - yc[1]) ** 2)); d12 = np.sqrt(np.mean((yc[1] - yc[2]) ** 2))
    return math.log2(d01 / d12)

for lin, ea, ns in [("picard", True, 1), ("newton", True, 1), ("picard", False, 1), ("newton", False, 3)]:
    order = centroid_temporal_order(lin, ea, ns)
    print(f"{lin:>7}, extrapolated_advection={ea!s:5}, subiter={ns}: centroid temporal order {order:.2f}")

# %% [markdown]
# | linearization | extrapolated advection | sub-iterations | centroid temporal order |
# |------|:---:|:---:|:---:|
# | picard | True | 1 | $1.905$ |
# | newton | True | 1 | $1.903$ |
# | picard | False | 1 | $1.035$ |
# | newton | False | 3 | $1.906$ |
#
# `extrapolated_advection=True` at `n_subiter=1` matches the old `n_subiter=3`
# baseline almost exactly — for **either** linearization, confirming the two
# flags are genuinely orthogonal — while `extrapolated_advection=False` at
# `n_subiter=1` reproduces the old first-order behaviour. This is why `run_case`
# now defaults to `linearization="picard"`, `extrapolated_advection=True`,
# `n_subiter=1`: the combined-refinement campaign above already uses it, at
# roughly a third of the cost of the equivalent `n_subiter=3` runs. The centroid
# error also drops by well over an order of magnitude:
#
# ![Temporal convergence of the centroid: extrapolated advection at n_subiter=1 is ~30x more accurate and second order.](hysing_turek/temporal_order.png)

# %% [markdown]
# ## Coarse meshes and the high-order (isoparametric) geometry
#
# `ngsxditto` never cuts high-order elements. It takes the $P^1$ interpolation of
# the levelset — which has unique, robust cut topologies — and restores accuracy
# with a high-order **mesh deformation**, the *isoparametric map*. The figure
# below shows this on the coarse **L0 mesh: 718 elements** with quadratic geometry
# (`order=2`). It overlays the straight-edged coarse cells with the curved interface
# the solver actually integrates on; the zoom contrasts the $P^1$ cut (straight
# chords) against the isoparametric curve.
#
# ![Coarse (order-2, 718-element) mesh with the isoparametric curved interface at three times, and a zoom comparing the P1 cut with the isoparametric curve.](hysing_turek/isoparametric_mapping.png)
#
# On this mesh the rise-velocity maximum already matches the reference to within
# about one percent ($v_{\max}\approx0.2444$ vs. the reference $0.2417$–$0.2421$),
# and the interface stays smooth as the cap forms and rises cleanly to $t=3$. High
# order buys **flow** accuracy on few cells; the interface still needs enough
# elements to keep the $P^1$ cuts non-degenerate.

# %% [markdown]
# ### The rising, deforming bubble
#
# The bubble accelerates from rest to a terminal rise velocity and deforms into
# the characteristic ellipsoidal cap. The storyboard and animation below show the
# curved (isoparametric) interface over the coarse ($h=0.08$, 718-element, order-2)
# background mesh across $t\in[0,3]$.
#
# ![Storyboard of the rising bubble on the coarse mesh: curved interface at t = 0, 0.6, 1.2, 1.8, 2.4, 3.0.](hysing_turek/storyboard.png)
#
# ![Animation of the rising bubble (coarse mesh, isoparametric interface).](hysing_turek/rising_bubble.gif)

# %% [markdown]
# ## Redistancing: keeping the level set a signed-distance function
#
# The level set lives in two complementary representations, which should not be
# mixed:
#
# * the **$P^1$ cut plus the isoparametric deformation** describes the *geometry*
#   — it is what the unfitted numerical integration sees (used for the interface,
#   curvature and storyboard figures above);
# * the **higher-order level-set field, without any deformation**, carries the
#   *signed-distance property* and is what the transport advects.
#
# By construction the two agree closely at the zero contour. Redistancing and
# $|\nabla\phi|$ are properties of the *second* representation, so the gallery
# below is drawn from the **higher-order field with no deformation applied**.
#
# The explicit-DG transport slowly distorts that field away from a signed-distance
# function ($|\nabla\phi|\neq1$), degrading the narrow band; the high-order
# variational `MinimizationBasedRedistancing` periodically pulls it back. The
# gallery shows its level sets at $\pm0.025\,i$ just **before** and just **after**
# several redistancing steps: the **zero** isoline — the interface, black — never
# moves, while the inner ($\phi<0$, blue) and outer ($\phi>0$, red) isolines,
# bunched where $|\nabla\phi|>1$ and spread where $|\nabla\phi|<1$, are restored to
# (nearly) even spacing.
#
# ![Before/after each redistancing: the ±0.025·i level sets over the mesh.](hysing_turek/redistancing_gallery.png)
#
# **When to redistance?** `PeriodicRedistancing(20)` fires every 20 steps — a
# fixed **12** times over $t\in[0,3]$ here (evenly, at $t=0.25,0.5,\dots,3.0$). A
# gradient-driven trigger (`GradientRedistancing`, band $[0.7,1.4]$: fire when
# $|\nabla\phi|$ leaves the band) is instead *adaptive* — it fired **18** times,
# but *later and where it matters*: the **first not until $t\approx0.56$** (the
# level set stays a clean signed-distance function while the bubble is still
# round), then roughly every $0.1$ once the cap deforms strongly. So a gradient
# trigger does **not** automatically redistance *less* — at this tolerance it
# redistances *more*, only reallocated to where $|\nabla\phi|$ actually drifts
# (arguably the smarter policy). To fire less often overall, one widens the band.

# %% [markdown]
# ## Mean curvature on the interface
#
# The surface-tension force is driven by the interface curvature from
# `MeanCurvatureSolver`. Below, the curved (isoparametric) interface is coloured
# by the magnitude of the mean curvature $|H|$ at several times: it is largest at
# the tightly rounded top and sides of the rising cap and smallest along the
# flattened underside — a direct look at what shapes the bubble.
#
# ![The isoparametric interface coloured by the mean-curvature magnitude.](hysing_turek/curvature_on_interface.png)

# %% [markdown]
# ## The velocity field
#
# Finally the flow itself, as streamlines over the speed with the interface on
# top. In the **laboratory frame** the classic rising-bubble pattern appears:
# fluid is pushed up through the bubble and returns down the side walls in two
# large recirculation cells, with the highest speeds at the shoulders of the
# widening cap.
#
# ![Velocity field (streamlines over speed) in the laboratory frame.](hysing_turek/velocity_absolute.png)
#
# Subtracting the mean bubble rise velocity $(0,\bar u_y)$ gives the flow **in the
# bubble's frame**, which exposes the **internal circulation** — two counter-
# rotating vortices inside the bubble, with the liquid sweeping around the cap.
#
# ![Velocity relative to the mean bubble rise velocity (bubble frame).](hysing_turek/velocity_relative.png)

# %% [markdown]
# ## Where the time goes
#
# Timing the L1 run ($h=0.04$, 2900 elements, to $t=3$, `n_subiter=1`) under a
# `TaskManager`, at the **module** level (`ngsxditto`'s own per-stepper timers),
# the two-phase flow solve still dominates:
#
# | module | wall time | share |
# |--------|:---:|:---:|
# | two-phase flow (`TwoPhaseTaylorHood`) | 1782 s | **70 %** |
# | levelset transport + redistancing + curvature (callback) | 389 s | 15 % |
# | velocity extension (`LevelsetBasedExtension`) | 215 s | 8 % |
# | mesh generation and setup | ≈150 s | 6 % |
# | QoI, snapshots, wind bookkeeping | 8 s | <1 % |
#
# (Mean curvature no longer appears as its own line: it now runs synchronously
# inside the level set's update callback — see *Second order in time* — so its
# cost is folded into the levelset figure above.)
#
# Within the fluid solve the split is **assembly** (convection + Stokes +
# time-stepping operators, $\approx610$ s, $34\%$) versus **linear algebra** (the
# sparse direct factorisation $\approx540$ s plus the BDF2 solves $\approx600$ s,
# $\approx1150$ s together, $64\%$) — the direct solve is still the single largest
# cost, and it is what makes the finest mesh so much slower.
#
# A short `TaskManager(pajetrace=…)` trace of the same setup gives the finer,
# **ngsolve-internal** picture below (grouped from the pajetrace timers): assembly of
# the cut bilinear forms, the Pardiso solve, the sparse matrix-graph setup, and the
# level-set **cut-geometry** work (`SymbolicCutBFI`, `LevelSetMeshAdaptation`) — the
# overhead the unfitted method adds over a fitted one. In this short trace the
# one-time assembly and graph-setup costs weigh heavily; over a full run the per-step
# solve grows to dominate, matching the module table above.
#
# ![Nested-pie breakdown of the ngsolve pajetrace timers: assembly, linear solve, matrix-graph setup and cut geometry.](hysing_turek/ngsolve_timers_sunburst.png)
#
# The interactive version below (self-contained, no external requests) lets you
# hover/click into each branch:
#
# <iframe src="_static/hysing_turek_pajetrace_sunburst.html" width="100%" height="560"
#         style="border:1px solid #ddd;border-radius:8px" title="pajetrace sunburst"></iframe>

# %% [markdown]
# ## Reproducing the figures
#
# Every figure above is produced by a small standalone script next to this
# notebook — each turns GridFunctions / CoefficientFunctions into a plot,
# headless (matplotlib + pyvista), so you can adapt them to your own runs:
#
# * `viz_common.py` — dump snapshots from a live `LevelSetGeometry`
#   (`dump_levelset`, `dump_curvature`) and read them back, warping the mesh by
#   the isoparametric deformation to recover the curved geometry;
# * `viz_isoparametric_interface.py` — coarse mesh + curved interface (+ $P^1$-vs-isoparametric zoom);
# * `viz_animation.py` — the rising-bubble storyboard and animated GIF;
# * `viz_redistancing_gallery.py` — the before/after level-set gallery;
# * `viz_curvature_on_interface.py` — the interface coloured by $|H|$;
# * `viz_qoi_convergence.py` — the QoI-vs-reference plot;
# * `viz_velocity_field.py` — the velocity field (laboratory and bubble frame).
#
# Each is importable and runnable as `python viz_<name>.py <data_dir> …`; the
# snapshots they consume are written with the `dump_*` helpers during a run (see
# `ht_redist_probe.py` for a fully instrumented example).

# %% [markdown]
# ## Summary
#
# * The `ngsxditto` unfitted two-phase solver reproduces the Hysing–Turek case-1
#   benchmark and **converges monotonically onto the reference range** under
#   refinement — the max. rise velocity goes $0.2444\to0.2425\to0.2419$, landing
#   inside the inter-group range $[0.2417,0.2421]$.
# * Thanks to the **high-order isoparametric geometry**, *very coarse* meshes
#   (a few hundred elements) already capture the interface accurately — a genuine
#   advantage of the CutFEM/isoparametric approach over low-order interface
#   tracking.
# * Key modelling choices for this inertial ($\mathrm{Re}=35$) case: convection
#   on (`add_convection=True`), BDF2 in time, and the body force passed as the
#   **acceleration** $\mathbf g$ (the solver multiplies by density).
# * The coupled flow ⇄ interface scheme is **second order in time with no Picard
#   sub-iteration** (`n_subiter=1`) once the convective linearization uses an
#   extrapolated, sub-iteration-refined predictor (`extrapolated_advection=True`)
#   and curvature/the velocity extension are timed consistently with the
#   just-updated geometry — see *Second order in time*. This is ~3–7$\times$
#   cheaper than the equivalent `n_subiter=3` runs at the same accuracy.
