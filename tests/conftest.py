from ngsolve import SetNumThreads
import pytest


@pytest.fixture(scope="session", autouse=True)
def _cap_ngsolve_threads():
    """Cap NGSolve's TaskManager thread pool for the whole test session.

    By default NGSolve uses all available cores. CI has intermittently
    aborted with a native SIGABRT deep in xfem.lsetcurv.CalcDeformation
    (see test_temporal_convergence.py::test_transport_wind_quadrature),
    which Solver's own num_threads docstring already flags as a real risk
    for "small problems (many small parallel kernels per step)" -- exactly
    the kind of problem most of this suite's tests use. Not confirmed as
    the actual cause (the crash is rare and wasn't reproducible locally
    either way), but a moderate, session-wide cap is cheap insurance and
    measurably speeds up the local suite (many-core dev machines otherwise
    oversubscribe these small meshes). Revisit if the crash recurs anyway.
    """
    SetNumThreads(4)
