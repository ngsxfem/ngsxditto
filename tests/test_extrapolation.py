"""Unit tests for Extrapolator (polynomial time extra-/interpolation)."""
import numpy as np
import pytest
from ngsolve import *
from ngsxditto.extrapolation import Extrapolator


@pytest.fixture(scope="module")
def fes():
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.3))
    return H1(mesh, order=1)


def _gf(fes, cf):
    g = GridFunction(fes)
    g.Set(cf)
    return g


def test_effective_order_startup_and_cap(fes):
    ex = Extrapolator(order=2)
    assert ex.EffectiveOrder() == 0
    for i, t in enumerate([0.0, 1.0, 2.0, 3.0, 4.0]):
        ex.Feed(t, _gf(fes, t * x))
        assert ex.EffectiveOrder() == min(i, 2)   # grows to `order`, then stays capped


@pytest.mark.parametrize("order", [0, 1, 2])
def test_reproduces_polynomial_of_matching_degree(fes, order):
    # An order-k extrapolator fed k+1 samples of a degree-k polynomial in time
    # reproduces it exactly, both inside (interpolation) and outside
    # (extrapolation) the fed time range.
    ex = Extrapolator(order=order)
    for t in range(order + 1):
        ex.Feed(float(t), _gf(fes, t**order * x))

    point = mesh_point = ex.gf.space.mesh(0.3, 0.4)
    for t_eval in [0.5 * order, order + 1.0]:   # interpolation, extrapolation
        got = ex.Evaluate(t_eval)(point)
        expected = t_eval**order * 0.3
        assert got == pytest.approx(expected, abs=1e-10)


def test_coincidence_check_scans_all_nodes_not_just_last(fes):
    # order=3 keeps up to 4 nodes; feed only 2, then re-feed near the FIRST
    # (not last) node's time.
    ex = Extrapolator(order=3)
    ex.Feed(0.0, _gf(fes, 0 * x))
    ex.Feed(1.0, _gf(fes, 1 * x))
    assert ex.EffectiveOrder() == 1

    ex.Feed(1e-12, _gf(fes, 42 * x))   # coincides with t=0, not t=1
    assert ex.EffectiveOrder() == 1    # still 2 nodes: overwritten, not appended

    point = ex.gf.space.mesh(1.0, 0.0)
    # the (interpolated) value at t=0.5 must reflect the overwritten node
    got = ex.Evaluate(0.5)(point)
    assert got == pytest.approx(0.5 * (42 + 1), abs=1e-8)


def test_non_chronological_feed_and_smallest_time_eviction(fes):
    ex = Extrapolator(order=1)   # ring size 2
    ex.Feed(5.0, _gf(fes, 5 * x))
    ex.Feed(1.0, _gf(fes, 1 * x))     # fed out of chronological order
    assert ex.EffectiveOrder() == 1

    ex.Feed(10.0, _gf(fes, 10 * x))   # buffer full -> evicts the smallest time (1.0)

    point = ex.gf.space.mesh(1.0, 0.0)
    got = ex.Evaluate(7.5)(point)       # midpoint of the surviving nodes (5, 10)
    assert got == pytest.approx(7.5, abs=1e-8)


def test_bare_vector_feed_needs_explicit_space_for_gf(fes):
    ex = Extrapolator(order=1)
    v0, v1 = _gf(fes, 0 * x).vec, _gf(fes, 1 * x).vec
    ex.Feed(0.0, v0)
    ex.Feed(1.0, v1)

    with pytest.raises(RuntimeError):
        ex.gf   # no space known yet

    out = ex.Evaluate(0.5)   # bare-vector evaluation works without a space
    assert isinstance(out, BaseVector)

    ex.SetSpace(fes)
    assert ex.gf is not None
    assert ex.vec is ex.gf.vec


def test_feeding_a_gridfunction_sets_space_automatically(fes):
    ex = Extrapolator(order=1)
    ex.Feed(0.0, _gf(fes, 0 * x).vec)   # bare vector first, no space yet
    ex.Feed(1.0, _gf(fes, 1 * x))       # GridFunction -> SetSpace called automatically
    assert ex.gf is not None


def test_evaluate_caches_until_time_changes_or_new_feed(fes):
    ex = Extrapolator(order=1)
    ex.Feed(0.0, _gf(fes, 0 * x))
    ex.Feed(1.0, _gf(fes, 1 * x))

    out = ex.Evaluate(0.5)
    out.vec.FV().NumPy()[:] = -999.0        # corrupt the cached output

    same_time_again = ex.Evaluate(0.5)      # same time -> cache hit, no recompute
    assert same_time_again.vec.FV().NumPy()[0] == pytest.approx(-999.0)

    different_time = ex.Evaluate(0.6)       # different time -> must recompute
    assert different_time.vec.FV().NumPy()[0] != pytest.approx(-999.0)

    different_time.vec.FV().NumPy()[:] = -999.0
    ex.Feed(2.0, _gf(fes, 2 * x))            # feeding invalidates the cache
    recomputed = ex.Evaluate(0.6)
    assert recomputed.vec.FV().NumPy()[0] != pytest.approx(-999.0)
