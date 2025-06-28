from ngsxditto.fluid import *
from ngsolve import *
import pytest


maxh = 0.1
mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
order = 4
fluid_params = FluidParameters(viscosity=1)
uin = CF((4 * y * (1 - y), 0))  # parabolic inflow


@pytest.mark.parametrize("fluid_type", [TaylorHood, ScottVogelius, BDMHDG])
def test_stokes(fluid_type):
    fluid = fluid_type(mesh, order=order, fluid_params=fluid_params)

    fluid.SetBoundaryConditions(dirichlet={"left|bottom|top": uin})
    fluid.InitializeSpaces(dbnd="left|bottom|top")
    fluid.InitializeForms(rhs=CF((1,0)))

    gfu = fluid.SolveStokes()

    l2_error = Integrate((gfu.components[0] - uin)**2, mesh)
    print("velocity L2-error: ", l2_error)
    assert l2_error < 1e-9
