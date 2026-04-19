from ngsxditto import *
from ngsolve import *
from xfem import *
from netgen.occ import *
from netgen.geom2d import SplineGeometry
import numpy as np


def test_levelset_convergence():
    circle_domain = MoveTo(0, 0).Circle(1).Face()
    dt = 2e-3
    end_time = 0.01
    h_list = [0.2, 0.1, 0.05]
    order_list = [1, 2, 3]

    t = Parameter(0)
    true_solution = ((x - 1 / 2 * sin(t)) ** 2 + (y + 1 / 2 * cos(t)) ** 2) ** (1 / 2) - 1 / 2
    wind = CF((-y, x))

    for order in order_list:
        max_error_list = []
        for h in h_list:
            t.Set(0)
            value_list = []
            mesh = Mesh(OCCGeometry(circle_domain, dim=2).GenerateMesh(maxh=h))
            transport = ExplicitDGTransport(mesh, dt=dt, order=order, wind=wind, compile=False)
            levelset = LevelSetGeometry(transport)
            levelset.Initialize(true_solution)
            phi_tilde = GridFunction(levelset.transport.fes)

            time_loop = TimeLoop(time=t, dt=dt, end_time=end_time, display_progress_bar=False, should_finalize=None,
                                 show_profiles=False)

            def append_to_value_list():
                phi_tilde.Set(shifted_eval(levelset.field, back=None, forth=levelset.deformation))

                diff = phi_tilde - true_solution
                interface_error = Integrate((diff * diff * levelset.dS), mesh=mesh) ** (1 / 2)
                value_list.append(interface_error)


            time_loop.Register(levelset)
            time_loop.Register(append_to_value_list)
            time_loop()

            max_error = np.max(value_list)
            max_error_list.append(max_error)

        p, logC = np.polyfit(np.log(np.array(h_list)), np.log(np.array(max_error_list)), 1)
        print(f"Estimated convergence order for order {order}: {p}")
        assert p >= (order + 1) - 0.2

def test_stokes_convergence():
    square_domain = MoveTo(-pi, -pi).Rectangle(2 * pi, 2 * pi).Face()
    nu = 0.1
    dt = 1e-5
    end_time = 10 * dt
    start_h = 0.8
    levels = 4
    h_list = [start_h * (2 ** (-i)) for i in range(levels)]
    print(h_list)
    order_list = [2, 3, 4]
    t = Parameter(0)
    true_velocity = CF((-cos(x) * sin(y) * exp(-2 * nu * t), sin(x) * cos(y) * exp(-2 * nu * t)))
    true_pressure = CF(-0.25 * (cos(2 * x) + cos(2 * y)) * exp(-4 * nu * t))
    r = pi / 2
    levelset_function = (x ** 2 + y ** 2) ** (1 / 2) - r
    grad_pressure = CF((sin(2 * x) * 0.5 * exp(-4 * nu * t), sin(2 * y) * 0.5 * exp(-4 * nu * t)))

    for order in order_list:
        max_error_list_u = []
        max_error_list_p = []
        mesh = Mesh(OCCGeometry(square_domain, dim=2).GenerateMesh(maxh=start_h))

        for i in range(levels):
            t.Set(0)
            levelset = LevelSetGeometry.from_cf(levelset_function, order=order, mesh=mesh)
            fluid_params = FluidParameters(viscosity=nu)
            fluid = TaylorHood(mesh, fluid_params, lset=levelset, order=order, dt=dt,
                               f=grad_pressure, add_convection=False, ghost_stab=1e-3, nitsche_stab=200,
                               extension_radius=0.2, add_number_space=True)
            fluid.SetInnerBoundaryCondition(true_velocity)
            fluid.Initialize(initial_velocity=true_velocity)

            value_list_u = [Integrate((fluid.gfu - true_velocity) ** 2 * levelset.dx_neg, mesh) ** (1 / 2)]
            value_list_p = []

            time_loop = TimeLoop(time=t, dt=dt, end_time=end_time, display_progress_bar=False, should_finalize=None,
                                 show_profiles=False)

            def append_to_value_list():
                diff_p = fluid.gfp - true_pressure
                mean_p = Integrate(diff_p * levelset.dx_neg, mesh) / Integrate(CF(1) * levelset.dx_neg, mesh)
                diff_p = (diff_p - mean_p)

                diff_u = fluid.gfu - true_velocity
                l2_error_u = Integrate(diff_u * diff_u * levelset.dx_neg, mesh=mesh) ** (1 / 2)
                value_list_u.append(l2_error_u)

                l2_error_p = Integrate(diff_p * diff_p * levelset.dx_neg, mesh=mesh) ** (1 / 2)
                value_list_p.append(l2_error_p)

            time_loop.Register(fluid)
            time_loop.Register(append_to_value_list)
            time_loop()

            max_error_u = np.max(value_list_u)
            max_error_list_u.append(max_error_u)

            max_error_p = np.max(value_list_p)
            max_error_list_p.append(max_error_p)
            mesh.Refine()

        p, logC = np.polyfit(np.log(np.array(h_list)), np.log(np.array(max_error_list_u)), 1)
        print(f"Estimated velocity convergence order for order {order}: {p}")
        assert p >= (order + 1) - 0.2

        p2, logC2 = np.polyfit(np.log(np.array(h_list)), np.log(np.array(max_error_list_p)), 1)
        print(f"Estimated pressure convergence order for order {order}: {p2}")
        assert p2 >= order - 0.2


def test_mean_curv_convergence():
    r = 0.5
    true_mean_curv = 1 / r
    true_normal = CF((2 * x, 2 * y))
    h_list = [0.2, 0.1, 0.05]
    order_list = [1, 2, 3]
    square_domain = MoveTo(-1, -1).Rectangle(2, 2).Face()

    levelset_function = (x ** 2 + y ** 2) ** (1 / 2) - r

    for order in order_list:
        error_list = []
        for h in h_list:
            mesh = Mesh(OCCGeometry(square_domain, dim=2).GenerateMesh(maxh=h))
            levelset = LevelSetGeometry.from_cf(levelset_function, mesh, order=order + 1)
            mean_curvature = MeanCurvatureSolver(mesh, order=order, lset=levelset, gp_param=specialcf.mesh_size ** (-2))
            mean_curvature.Step()
            norm = Norm(mean_curvature.H)
            error = Integrate(Norm(mean_curvature.H - true_mean_curv * true_normal) ** 2 * levelset.dS,
                              mesh) ** (1 / 2)
            error_list.append(error)
        p, logC = np.polyfit(np.log(np.array(h_list)), np.log(np.array(error_list)), 1)
        print(f"Estimated convergence order for order {order}: {p}")
        assert p >= order
