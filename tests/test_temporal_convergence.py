"""Temporal-order (dt-refinement) tests at fixed mesh.

Self-convergence errors (finest run as reference on the SAME mesh/space) are
used for the order assertions -- they cancel the spatial error floor exactly.

Pinned behavior:
 * BDF2 startup: the first validated step is backward Euler with FULL dt,
   so `time_order=2` is genuinely 2nd order (it used to degenerate to BE
   with an effective step of (2/3)dt, capping the global order at 1).
 * Transport wind quadrature: the explicit-DG transport freezes the wind
   during a step; with the endpoint wind it is 1st order, with an order-1
   Extrapolator evaluated at the interval midpoint (a 2nd-order predictor of
   the mid-step wind from the validated history) it is 2nd order.
"""
import numpy as np
import pytest
from ngsxditto import *
from ngsolve import *
from xfem import *
from netgen.occ import *


def _lsq_order(dts, errs):
    p, _ = np.polyfit(np.log(np.array(dts)), np.log(np.array(errs)), 1)
    return p


# ---------------------------------------------------------------------------
# fixed cut domain, unsteady Stokes MMS (BDF2 startup)
# ---------------------------------------------------------------------------
def _run_stokes_series(time_order, dts, dt_ref, maxh=0.5, order=2, t_end=0.25):
    nu, w_freq = 0.1, 2*pi
    square = MoveTo(-pi, -pi).Rectangle(2*pi, 2*pi).Face()
    mesh = Mesh(OCCGeometry(square, dim=2).GenerateMesh(maxh=maxh))
    lset_cf = (x**2 + y**2)**(1/2) - pi/2

    t = Parameter(0)
    g = cos(w_freq*t) + sin(w_freq*t)          # g'(0) != 0: startup error visible
    gp = w_freq*(-sin(w_freq*t) + cos(w_freq*t))
    u_sp = CF((-cos(x)*sin(y), sin(x)*cos(y)))
    true_velocity = g * u_sp
    rhs_f = (gp + 2*nu*g) * u_sp + g * CF((0.5*sin(2*x), 0.5*sin(2*y)))

    V_store = VectorH1(mesh, order=order)

    def run(dt):
        t.Set(0)
        levelset = LevelSetGeometry.from_cf(lset_cf, order=order, mesh=mesh)
        fluid = TaylorHood(mesh, FluidParameters(viscosity=nu), lset=levelset,
                           order=order, dt=dt, f=rhs_f, add_convection=False,
                           ghost_stab=1e-3, nitsche_stab=200, extension_radius=0.2,
                           add_number_space=True, time_order=time_order)
        fluid.SetInnerBoundaryCondition(true_velocity)
        fluid.Initialize(initial_velocity=true_velocity)
        loop = TimeLoop(time=t, dt=dt, end_time=t_end,
                        display_progress_bar=False, show_profiles=False)
        loop.Register(fluid)
        loop()
        uf = GridFunction(V_store); uf.vec.data = fluid.gfu.vec
        return uf, levelset

    finals = [run(dt)[0] for dt in dts]
    u_ref, lset = run(dt_ref)
    diff = GridFunction(V_store)
    errs = []
    for uf in finals:
        diff.vec.data = uf.vec - u_ref.vec
        errs.append(Integrate(diff**2 * lset.dx_neg, mesh)**0.5)
    return _lsq_order(dts, errs)


def test_bdf2_startup_fixed_domain():
    dts = [0.25/8, 0.25/16, 0.25/32]
    p2 = _run_stokes_series(time_order=2, dts=dts, dt_ref=0.25/128)
    print(f"time_order=2: observed temporal order p = {p2:.2f}")
    assert p2 > 1.7, f"BDF2 lost 2nd order (p={p2:.2f}) - startup regression?"

    p1 = _run_stokes_series(time_order=1, dts=dts, dt_ref=0.25/128)
    print(f"time_order=1: observed temporal order p = {p1:.2f}")
    assert p1 < 1.4, f"BE control unexpectedly 2nd order (p={p1:.2f})"


# ---------------------------------------------------------------------------
# transport alone with analytic time-dependent wind (wind quadrature)
# ---------------------------------------------------------------------------
def _run_transport_series(mode, dts, subs, dt_ref, sub_ref,
                          maxh=0.15, order=2, t_end=0.5):
    disk = MoveTo(0, 0).Circle(1).Face()
    mesh = Mesh(OCCGeometry(disk, dim=2).GenerateMesh(maxh=maxh))
    t = Parameter(0)
    G = sin(t)
    phi_exact = ((x - 0.5*cos(G))**2 + (y - 0.5*sin(G))**2)**0.5 - 0.3
    wind_cf = cos(t) * CF((-y, x))

    def run(dt, substeps):
        t.Set(0)
        transport = ExplicitDGTransport(mesh, dt=dt, order=order, compile=False,
                                        substeps=substeps)
        levelset = LevelSetGeometry(transport)
        levelset.Initialize(phi_exact)
        loop = TimeLoop(time=t, dt=dt, end_time=t_end,
                        display_progress_bar=False, show_profiles=False)
        if mode == "endpoint":
            # frozen endpoint wind w^{n+1} -> 1st order
            transport.SetWind(wind_cf)
            loop.Register(levelset)
        else:
            # order-1 Extrapolator: midpoint wind extrapolated from the
            # validated history {w^{n-1}, w^n} -> 2nd order. No live producer
            # here (analytic wind), so we feed a snapshot ourselves at each
            # validated step (only_on_validate behaviour).
            wgf = GridFunction(VectorH1(mesh, order=order))
            wgf.Set(wind_cf)                       # w^0 at t=0
            wind = Extrapolator(order=1)
            wind.Feed(0.0, wgf)                    # prime the history
            transport.SetWind(wind.gf)

            def feed_wind():                       # t == t^{n+1} at validate
                wgf.Set(wind_cf)
                wind.Feed(t.Get(), wgf)

            loop.Register(lambda: wind.Evaluate(t.Get() - dt/2), name="wind")
            loop.Register(levelset)
            loop.Register(feed_wind, name="wind feed", as_validate=True)
        loop()
        phi_tilde = GridFunction(levelset.transport.fes)
        phi_tilde.Set(shifted_eval(levelset.field, back=None,
                                   forth=levelset.deformation))
        return phi_tilde, levelset

    finals = [run(dt, s)[0] for dt, s in zip(dts, subs)]
    p_ref, lset = run(dt_ref, sub_ref)
    errs = []
    for pf in finals:
        d = GridFunction(p_ref.space); d.vec.data = pf.vec - p_ref.vec
        errs.append(Integrate(d**2 * lset.dS, mesh)**0.5)
    return _lsq_order(dts, errs)


def test_transport_wind_quadrature():
    # keep dt_sub = dt/substeps identical across ALL runs (incl. reference) so
    # the pseudo-time RK integration error cancels in the self-convergence.
    dts, subs = [0.1, 0.05, 0.025], [16, 8, 4]
    dt_ref, sub_ref = 0.025/4, 1
    p_end = _run_transport_series("endpoint", dts, subs, dt_ref, sub_ref)
    print(f"endpoint wind:      observed order p = {p_end:.2f}")
    assert p_end < 1.4, f"endpoint wind unexpectedly 2nd order (p={p_end:.2f})"

    p_extr = _run_transport_series("extrapolator", dts, subs, dt_ref, sub_ref)
    print(f"extrapolated wind:  observed order p = {p_extr:.2f}")
    assert p_extr > 1.7, f"midpoint Extrapolator lost 2nd order (p={p_extr:.2f})"


# ---------------------------------------------------------------------------
# moving domain with prescribed levelset motion (Eulerian BDF2 + history)
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_moving_domain_bdf2():
    nu, a_amp, w_freq = 0.1, 0.5, 2.0
    maxh, order, t_end = 0.15, 3, 0.5
    square = MoveTo(-1, -1).Rectangle(2, 2).Face()
    mesh = Mesh(OCCGeometry(square, dim=2).GenerateMesh(maxh=maxh))

    t = Parameter(0)
    c = 0.25*sin(t); cdot = 0.25*cos(t); cddot = -0.25*sin(t)
    phi_exact = ((x - c)**2 + y**2)**0.5 - 0.5
    g = cos(w_freq*t) + sin(w_freq*t)
    gp = w_freq*(-sin(w_freq*t) + cos(w_freq*t))
    # rotational near the origin (~(-y,x)); a saddle-like/gradient field here
    # would be pressure-absorbed and hide all temporal error
    u_sp = CF((-cos(x)*sin(y), sin(x)*cos(y)))
    true_velocity = CF((cdot, 0)) + a_amp*g*u_sp
    rhs_f = CF((cddot, 0)) + a_amp*(gp + 2*nu*g)*u_sp

    V_store = VectorH1(mesh, order=order)

    def run(dt):
        t.Set(0)
        transport = KnownSolutionTransport(mesh, phi_exact, time=t, dt=dt, order=order)
        levelset = LevelSetGeometry(transport)
        levelset.Initialize(phi_exact)
        fluid = TaylorHood(mesh, FluidParameters(viscosity=nu), lset=levelset,
                           order=order, dt=dt, f=rhs_f, add_convection=False,
                           ghost_stab=1e-3, nitsche_stab=200, extension_radius=0.2,
                           add_number_space=True, time_order=2)
        fluid.SetInnerBoundaryCondition(true_velocity)
        fluid.Initialize(initial_velocity=true_velocity)
        loop = TimeLoop(time=t, dt=dt, end_time=t_end,
                        display_progress_bar=False, show_profiles=False)
        loop.Register(levelset, name="levelset")
        loop.Register(fluid, name="fluid")
        loop()
        uf = GridFunction(V_store); uf.vec.data = fluid.gfu.vec
        return uf, levelset

    dts = [0.5/8, 0.5/16, 0.5/32]
    finals = [run(dt)[0] for dt in dts]
    u_ref, lset = run(0.5/128)
    diff = GridFunction(V_store)
    errs = []
    for uf in finals:
        diff.vec.data = uf.vec - u_ref.vec
        errs.append(Integrate(diff**2 * lset.dx_neg, mesh)**0.5)
    p = _lsq_order(dts, errs)
    print(f"moving-domain BDF2: observed temporal order p = {p:.2f}")
    assert p > 1.7, f"moving-domain BDF2 lost 2nd order (p={p:.2f})"
