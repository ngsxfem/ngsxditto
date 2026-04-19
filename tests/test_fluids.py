from ngsxditto.fluid import *
from ngsolve import *
import pytest



domain = MoveTo(-1, -1).Rectangle(2, 2).Face()
domain.edges.Max(X).name = "right"
domain.edges.Min(X).name = "left"
domain.edges.Min(Y).name = "bottom"
domain.edges.Max(Y).name = "top"
mesh = Mesh(OCCGeometry(domain, dim=2).GenerateMesh(maxh=0.1))

order = 4
nu = 1
fluid_params = FluidParameters(viscosity=nu)

true_solution_u = CF((pi * cos(pi * x) * sin(pi * y), -pi * sin(pi * x) * cos(pi * y)))
true_solution_p = CF(cos(pi * x) * cos(pi * y))

f = CF((pi * (2 * pi ** 2 * nu * sin(pi * y) * cos(pi * x) - sin(pi * x) * cos(pi * y)),
          -pi * (2 * pi ** 2 * nu * sin(pi * x) * cos(pi * y) + sin(pi * y) * cos(pi * x))))


@pytest.mark.parametrize("fluid_type", [TaylorHood])
def test_fitted_stokes(fluid_type):
    fluid = fluid_type(mesh, order=order, fluid_params=fluid_params, f=f, add_number_space=True)
    fluid.SetOuterBoundaryCondition(StrongDirichletBC(region="top|bottom|right|left", values=true_solution_u))
    fluid.Initialize()

    sol = fluid.SolveStokes()
    uh, ph, _ = sol.components
    fluid.SetInitialValues(uh, ph)

    l2_error_u = Integrate((fluid.gfu - true_solution_u)**2, mesh)
    assert l2_error_u < 1e-3
    p_error = fluid.gfp - true_solution_p
    vol = 4
    average_diff = 1 / vol * Integrate(p_error, mesh)
    corrected_p_error = fluid.gfp - average_diff - true_solution_p
    l2_error_p = Integrate(corrected_p_error ** 2 * fluid.lset.dx_neg, mesh) ** (1/2)
    assert l2_error_p < 1e-3


@pytest.mark.parametrize("fluid_type", [TaylorHood])
def test_unfitted_stokes(fluid_type):
    levelset_function = (x**2 + y**2) ** (1/2) - 0.5
    transport = KnownSolutionTransport(mesh, levelset_function, order=2)
    levelset = LevelSetGeometry(transport)
    levelset.Initialize(levelset_function)

    fluid = fluid_type(mesh, fluid_params, f=f, lset=levelset, order=order,
                       ghost_stab=1, nitsche_stab=100, add_number_space=True)


    fluid.Initialize()
    sol = fluid.SolveStokes()
    uh, ph, _ = sol.components
    fluid.SetInitialValues(uh, ph)

    u_error = fluid.gfu - true_solution_u
    l2_error_u = Integrate(InnerProduct(u_error,u_error) * fluid.lset.dx_neg, mesh)**(1/2)
    assert l2_error_u < 1e-3

    p_error = fluid.gfp - true_solution_p

    vol = Integrate(CF(1) * fluid.lset.dx_neg, mesh)
    average_diff = 1 / vol * Integrate(p_error * fluid.lset.dx_neg, mesh)
    corrected_p_error = fluid.gfp - average_diff - true_solution_p
    l2_error_p = Integrate(corrected_p_error ** 2 * fluid.lset.dx_neg, mesh) ** (1 / 2)
    assert l2_error_p < 1e-3
