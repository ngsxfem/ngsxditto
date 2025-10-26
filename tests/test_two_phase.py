from ngsxditto import *
from ngsolve import *



domain = MoveTo(-1, -1).Rectangle(2, 2).Face()
domain.edges.Max(X).name = "right"
domain.edges.Min(X).name = "left"
domain.edges.Min(Y).name = "bottom"
domain.edges.Max(Y).name = "top"
mesh = Mesh(OCCGeometry(domain, dim=2).GenerateMesh(maxh=0.15))

order = 4

# 
nu1 = 1
nu2 = 1e-1

fluid1_params = FluidParameters(viscosity=nu1)
fluid2_params = FluidParameters(viscosity=nu2)

true_solution_u = CF(( y**2, 0.0 ))
true_solution_p = CF( x )

# piecewise forcings for -nu * Delta u + grad p = f
# note: Delta u = (2, 0), grad p = (1,0) => f_i = (1 - 2*nu_i, 0)
f1 = CF(( 1.0 - 2.0 * nu1, 0.0 ))
f2 = CF(( 1.0 - 2.0 * nu2, 0.0 ))

levelset_function = CF(y)

def test_two_phase_stokes():
    dirichlet = {"left|right|bottom|top": true_solution_u}
    levelset = LevelSetGeometry.from_cf(levelset_function, mesh, order=2)

    fluid = TwoPhaseTaylorHood(mesh, fluid1_params, fluid2_params, order=order, lset=levelset,
                               f1=f1, f2=f2, dt=0.1, ghost_stab=0)
    fluid.Initialize(dirichlet=dirichlet)
    sol = fluid.SolveStokes()
    u1, p1, u2, p2 = sol.components
    fluid.SetInitialValues(u1, u2, p1, p2)

    u_error1 = fluid.gfu.components[0] - true_solution_u
    u_error2 = fluid.gfu.components[2] - true_solution_u

    u_error_total = IfPos(fluid.lset.lsetp1, u_error2, u_error1)

    l2_error_u = Integrate(InnerProduct(u_error_total,u_error_total) * dx, mesh)**(1/2)

    assert l2_error_u < 1e-10

    p_error1 = fluid.gfu.components[1] - true_solution_p
    p_error2 = fluid.gfu.components[3] - true_solution_p

    p_error_total = IfPos(fluid.lset.lsetp1, p_error2, p_error1)

    vol = 4
    average_diff = 1 / vol * Integrate(p_error_total, mesh)
    corrected_p_error = IfPos(fluid.lset.lsetp1, fluid.gfu.components[3],  fluid.gfu.components[1]) - average_diff - true_solution_p

    l2_error_p = Integrate(InnerProduct(corrected_p_error, corrected_p_error) * dx, mesh)**(1/2)

    assert l2_error_p < 1e-10
