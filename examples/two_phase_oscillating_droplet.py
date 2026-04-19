# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from ngsxditto import *
from ngsolve import *
from xfem import *
import ngsolve.webgui as ngw
from netgen.occ import *

# %%
domain = MoveTo(-1, -1).Rectangle(2, 2).Face()
domain.edges.Max(X).name = "right"
domain.edges.Min(X).name = "left"
domain.edges.Min(Y).name = "bottom"
domain.edges.Max(Y).name = "top"
mesh = Mesh(OCCGeometry(domain, dim=2).GenerateMesh(maxh=0.15))

# %%
dt = 0.05
order = 2
t = Parameter(0)
starting_levelset = (5*x**2 + y**2)**(1/2) - 2.0/3.0
transport = ExplicitDGTransport(mesh, dt=dt, order=order, compile=False)
levelset = LevelSetGeometry(transport)
levelset.Initialize(starting_levelset)

# %%
fluid1_params = FluidParameters(viscosity=1e-2, surface_tension_coeff=0.2)
fluid2_params = FluidParameters(viscosity=1e-3)

mean_curvature = MeanCurvatureSolver(mesh, order=order, lset=levelset)
mean_curvature.Step()
fluid = TwoPhaseTaylorHood(mesh, fluid1_params=fluid1_params, fluid2_params=fluid2_params, lset=levelset, surface_tension=mean_curvature.H, dt=dt, order=order + 1,
                           ghost_stab=1, nitsche_stab=100)
fluid.Initialize(dirichlet={".*": CF((0, 0))})
fluid.ValidateStep()
sol = fluid.SolveStokes()
gfu, gfp = sol.components[0], sol.components[1]
u_neg, u_pos = gfu.components
p_neg, p_pos = gfp.components
ngw.Draw(IfPos(levelset.lsetp1, u_pos, u_neg), mesh, "u")
ngw.Draw(IfPos(levelset.lsetp1, p_pos, p_neg), mesh, "u")

# %%
velocity_extension = LevelsetBasedExtension(levelset, gamma=1e-3, order=order)

velocity_extension.SetRhs(fluid.gfu.components[0])
levelset.transport.SetWind(velocity_extension.field)

end_time = 2

time_loop = TimeLoop(time=t, dt=dt, end_time=end_time)

sphericity = SphericityDiagram(levelset, time=t, name="sphericity")

cf_neg = Norm(fluid.gfu.components[0])
cf_pos = Norm(fluid.gfu.components[1])
animation = UnfittedNGSWebguiPlot(levelset, cf_neg=cf_neg, cf_pos=cf_pos,
                                  order=fluid.order, time=t, end_time=end_time,
                                  name="animation", min=0, max=0.5, autoscale=False)

time_loop.Register(fluid, name="moving stokes")
time_loop.Register(velocity_extension, name="vel ext.")
time_loop.Register(levelset, name="levelset")
time_loop.Register(mean_curvature, name="mean curvature")

time_loop.Register(sphericity, name="sphericity")
time_loop.Register(animation, name="animation")
time_loop()
