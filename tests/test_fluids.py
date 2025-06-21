from ngsxditto.fluid import TaylorHood, FluidParameters
from ngsolve import *


def test_h1conf_import():
    maxh = 0.2
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fluid_params = FluidParameters(viscosity=1e-3)
    fluid = TaylorHood(mesh, fluid_params=fluid_params)


def test_h1conf_stokessolution():
    maxh = 0.1
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))

    order = 4

    fluid_params = FluidParameters(viscosity=1)
    fluid = TaylorHood(mesh, order=order, fluid_params=fluid_params)

    uin = CF((4*y*(1-y),0)) # parabolic inflow
    fluid.SetBoundaryConditions(dirichlet={"left|bottom|top": uin})
    fluid.InitializeSpaces(dbnd="left|bottom|top")
    fluid.InitializeForms(rhs=CF((1,0)))

    gfu = fluid.SolveStokes()

    fes = VectorH1(mesh, order=order)
    gfu_exact = GridFunction(fes)
    gfu_exact.Set(uin) # interpolated exact solution

    l2_error = Integrate((gfu.components[0] - gfu_exact)**2, mesh)
    print("velocity L2-error: ", l2_error, " (compared to best approximation)")

    assert l2_error < 1e-9 
