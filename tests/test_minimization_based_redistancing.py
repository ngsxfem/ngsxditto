from netgen.libngpy._meshing import Mesh

from ngsxditto import LevelSetGeometry
from ngsxditto.redistancing import *
from ngsolve import *
from netgen.geom2d import SplineGeometry
from xfem.lsetcurv import *


domain = SplineGeometry()
domain.AddCircle((0,0), 1)
mesh = Mesh(domain.GenerateMesh(maxh=0.1))
circle = x**2 + y**2 - 0.25
true_signed_distance = (x**2 + y**2)**(1/2) - 1/2
domain_size = Integrate(CF(1), mesh)

def test_low_order():
    order = 1

    redistancing = MinimizationBasedRedistancing(alpha=10000)
    levelset = LevelSetGeometry.from_cf(circle, mesh, order=order)
    levelset.SetRedistancing(redistancing)
    levelset.Redistance()
    d_hasif = dx(definedonelements=levelset.cutinfo.GetElementsOfType(IF))
    hasif_size = Integrate(CF(1) * d_hasif, mesh)


    assert 1/domain_size * Integrate((levelset.field - true_signed_distance)**2, mesh)**(1/2) < 1e-2  # L2-error
    assert 1/hasif_size * Integrate((levelset.field - true_signed_distance)**2 * d_hasif, mesh)**(1/2) < 1e-2  # hasif l2 error
    assert Integrate(true_signed_distance**2 * levelset.dS, mesh)**(1/2) < 1e-2  # check preserves interface
    assert Integrate((Norm(grad(levelset.field)) - CF(1))**2, mesh)**(1/2) < 1e-1  # gradient error

def test_high_order():
    order = 3

    redistancing = MinimizationBasedRedistancing(alpha=10000, n_iter=10)
    levelset = LevelSetGeometry.from_cf(circle, mesh, order=order)
    levelset.SetRedistancing(redistancing)
    levelset.Redistance()
    d_hasif = dx(definedonelements=levelset.cutinfo.GetElementsOfType(IF))
    hasif_size = Integrate(CF(1) * d_hasif, mesh)

    assert 1/domain_size * Integrate((levelset.field - true_signed_distance)**2, mesh)**(1/2) < 1e-3  # L2-error
    assert 1/hasif_size * Integrate((levelset.field - true_signed_distance)**2 * d_hasif, mesh)**(1/2) < 1e-3  # hasif l2 error
    assert Integrate(true_signed_distance**2 * levelset.dS, mesh)**(1/2) < 1e-5  # interface error
    assert 1/hasif_size * Integrate((Norm(grad(levelset.field)) - CF(1))**2 * d_hasif, mesh)**(1/2) < 1e-2  # gradient error near interface


# a rough/noisy input (mimicking a level set drifted by many transport steps)
# to actually exercise the initializer= hook: on the plain quadratic `circle`
# field there is nothing for an initializer to meaningfully improve on.
rough_circle = circle + 0.05 * sin(15 * x) * cos(13 * y)


def _l2_err(field):
    return (1 / domain_size * Integrate((field - true_signed_distance) ** 2, mesh)) ** (1 / 2)


def _grad_rms(field):
    return (1 / domain_size * Integrate((Norm(grad(field)) - CF(1)) ** 2, mesh)) ** (1 / 2)


def test_initializer_fastmarching_improves_on_rough_input():
    order = 2
    baseline = LevelSetGeometry.from_cf(rough_circle, mesh, order=order)
    baseline.SetRedistancing(MinimizationBasedRedistancing(alpha=10000, n_iter=10))
    baseline.Redistance()

    initialized = LevelSetGeometry.from_cf(rough_circle, mesh, order=order)
    initialized.SetRedistancing(MinimizationBasedRedistancing(alpha=10000, n_iter=10, initializer=FastMarching()))
    initialized.Redistance()

    assert _l2_err(initialized.field) < _l2_err(baseline.field)
    assert _grad_rms(initialized.field) < _grad_rms(baseline.field)
    assert _l2_err(initialized.field) < 5e-2
    assert _grad_rms(initialized.field) < 1e-1


def test_initializer_gp_extension_improves_on_rough_input():
    order = 2
    baseline = LevelSetGeometry.from_cf(rough_circle, mesh, order=order)
    baseline.SetRedistancing(MinimizationBasedRedistancing(alpha=10000, n_iter=10))
    baseline.Redistance()

    initialized = LevelSetGeometry.from_cf(rough_circle, mesh, order=order)
    initialized.SetRedistancing(
        MinimizationBasedRedistancing(alpha=10000, n_iter=10, initializer=GPExtensionRedistancing()))
    initialized.Redistance()

    assert _l2_err(initialized.field) < _l2_err(baseline.field)
    assert _grad_rms(initialized.field) < _grad_rms(baseline.field)
    assert _l2_err(initialized.field) < 5e-2
    assert _grad_rms(initialized.field) < 1e-1


# --- interface-penalty scaling -------------------------------------------------
# The interface is held in place by a penalty, so its position is preserved only
# to O(1/alpha). The volume term of the energy is O(1) per element while an
# unscaled interface term scales like alpha*h, so a FIXED alpha gets relatively
# weaker under refinement and the drift of the zero set stops converging. The
# default therefore scales as alpha_scale/h**2.

_A, _B, _CX, _CY = 0.32, 0.18, 0.5, 0.75


def _drift_after_one_pass(maxh, redistancer, nray=360):
    """Move a level set whose zero set is *analytically* unchanged, redistance it
    once, and measure how far the zero contour actually moved.

    The start field is an exact ellipse times a strictly positive factor: the
    zero set is identical, only |grad phi| is perturbed -- the same kind of
    defect a transported level set accumulates between redistancing passes.
    """
    import numpy as np
    geo = SplineGeometry()
    geo.AddRectangle((0, 0), (1, 2), bcs=("bottom", "right", "top", "left"))
    m = Mesh(geo.GenerateMesh(maxh=maxh))
    exact = (((x - _CX) / _A)**2 + ((y - _CY) / _B)**2)**0.5 - 1
    lset = LevelSetGeometry.from_cf(
        exact * (1 + 0.15 * sin(6 * atan2(y - _CY, x - _CX))), m, 2)
    lset.SetRedistancing(redistancer)

    def radii():
        th = np.linspace(0, 2 * np.pi, nray, endpoint=False)
        lo, hi = np.full(nray, 0.02), np.full(nray, 0.9)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            v = np.array(lset.field(m(_CX + mid * np.cos(th),
                                      _CY + mid * np.sin(th)))).ravel()
            neg = v < 0
            lo, hi = np.where(neg, mid, lo), np.where(neg, hi, mid)
        return 0.5 * (lo + hi)

    before = radii()
    lset.Redistance()
    return float(np.sqrt(np.mean((radii() - before)**2)))


def test_zero_set_drift_converges_with_default_penalty():
    coarse = _drift_after_one_pass(0.04, MinimizationBasedRedistancing())
    fine = _drift_after_one_pass(0.02, MinimizationBasedRedistancing())
    # measured ~6x (order 2.6); a fixed alpha=1e4 gives only ~1.5x (order 0.5)
    assert fine < coarse / 3.0, (
        f"zero-set drift barely improved under refinement: {coarse:.3e} -> {fine:.3e}; "
        "the interface penalty is probably not scaling with the mesh")


def test_pinned_alpha_is_left_unscaled():
    """An explicitly given alpha keeps its old, absolute meaning."""
    r = MinimizationBasedRedistancing(alpha=10000)
    assert r.alpha == 10000
    assert MinimizationBasedRedistancing().alpha is None
