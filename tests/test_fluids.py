from ngsxditto.fluid import H1ConformingFluid, FluidParameters
from ngsolve import *


def test_h1conf_import():
    maxh = 0.2
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fluid_params = FluidParameters(viscosity=1e-3)
    fluid = H1ConformingFluid(mesh, fluid_params=fluid_params)


def test_h1conf_stokessolution():
    maxh = 0.1
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))

    order = 4

    fluid_params = FluidParameters(viscosity=1)
    fluid = H1ConformingFluid(mesh, order=order, fluid_params=fluid_params)

    uin = CF((4*y*(1-y),0)) # parabolic inflow
    fluid.SetBoundaryCondition(Dbndc=uin, Dbnd="left|bottom|top")
    fluid.InitializeVarForm(rhs=CF((1,0)))

    gfu = fluid.SolveStokes()

    fes = VectorH1(mesh, order=order)
    gfu_exact = GridFunction(fes)
    gfu_exact.Set(uin) # interpolated exact solution

    Draw(gfu.components[0])

    l2_error = Integrate((gfu.components[0] - gfu_exact)**2, mesh)
    print("velocity L2-error: ", l2_error, " (compared to best approximation)")

    assert l2_error < 1e-9 


if __name__=="__main__":
    test_h1conf_import()
    test_h1conf_stokessolution()
