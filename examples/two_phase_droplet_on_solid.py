# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Two Phase droplet on a solid surface

# %%
from ngsxditto.utils.loglevel import loggingSlider
loggingSlider(default_level="WARNING") # global level
loggingSlider("ngsxditto", default_level="DEBUG") # ngsxditto

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
dt = 2e-2
order = 2
t = Parameter(0)
starting_levelset = (x**2 + (y + 0.75)**2)**0.5 - 1/2
transport = ExplicitDGTransport(mesh, dt=dt, order=order, compile=False)
levelset = LevelSetGeometry(transport)
levelset.Initialize(starting_levelset)
ngw.Draw(levelset.field)

# %%
fluid1_params = FluidParameters(viscosity=1e-1, surface_tension_coeff=1)
fluid2_params = FluidParameters(viscosity=1e-2)

wall_params = WallParameters(region="bottom", contact_angle=pi/3, friction_coeff_surface=1)
mean_curvature = MeanCurvatureSolver(mesh, order=order, lset=levelset)
mean_curvature.Step()
fluid = TwoPhaseTaylorHood(mesh, fluid1_params=fluid1_params, fluid2_params=fluid2_params,
                           lset=levelset, nitsche_stab=100, f1=CF((0, -9.8)), f2=CF((0, -9.8)),
                           surface_tension=mean_curvature.H, dt=dt, order=order + 1, ghost_stab=1e-2,
                           add_convection=True, add_number_space=False, time_order=1,
                           wall_params=wall_params)
fluid.SetOuterBoundaryCondition(NitscheVelocityBC(region="right|left|top", values=CF((0, 0))))
fluid.SetOuterBoundaryCondition(NitscheNormalVelocityBC(region="bottom", values=CF(0)))
fluid.Initialize()

sol = fluid.SolveStokes()
gfu, gfp = sol.components[0], sol.components[1]
u1, u2 = gfu.components
p1, p2 = gfp.components

DrawDC(levelset.field, u1, u2, mesh)

# %%
velocity_extension = LevelsetBasedExtension(levelset, order=order, gamma=1e-1, ghost_stab=1, no_slip="top|left|right")

velocity_extension.SetRhs(fluid.gfu.components[0])
levelset.transport.SetWind(velocity_extension.field)

def should_finalize():
    return time_loop.i_inner == 3

end_time = 2

time_loop = TimeLoop(time=t, dt=dt, end_time=end_time, display_progress_bar=True, should_finalize=None)
time_loop.SetFinalizeRule(should_finalize)

cf_neg = Norm(fluid.gfu.components[0])
cf_pos = Norm(fluid.gfu.components[1])
animation = UnfittedNGSWebguiPlot(levelset, cf_neg=cf_neg, cf_pos=cf_pos,
                                  order=fluid.order, time=t, end_time=end_time,
                                  name="animation", min=-0.075, max=0.4, autoscale=False)

time_loop.Register(velocity_extension, name="vel ext.")
time_loop.Register(levelset, name="levelset")
time_loop.Register(mean_curvature, name="mean curvature")
time_loop.Register(fluid, name="moving stokes")
time_loop.Register(animation, name="animation")

time_loop()

# %%
