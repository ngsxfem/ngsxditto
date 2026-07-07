# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Droplet on a solid surface

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
maxh = 0.15
domain = MoveTo(-1, -1).Rectangle(2, 2).Face()
domain.edges.Max(X).name = "right"
domain.edges.Min(X).name = "left"
domain.edges.Min(Y).name = "bottom"
domain.edges.Max(Y).name = "top"
domain.edges.Min(Y).maxh = 0.4 * maxh
mesh = Mesh(OCCGeometry(domain, dim=2).GenerateMesh(maxh=maxh))
ngw.Draw(mesh)

# %%
dt = 2e-2
order = 1
t = Parameter(0)
starting_levelset = (x**2 + (y + 0.75)**2)**0.5 - 0.5
transport = ExplicitDGTransport(mesh, dt=dt, order=order, compile=False)
levelset = LevelSetGeometry(transport)
levelset.Initialize(starting_levelset)
ngw.Draw(levelset.field)

# %%
fluid_params = FluidParameters(viscosity=5e-2, surface_tension_coeff=1)
wall_params = WallParameters(region="bottom", contact_angle=pi/2, friction_coeff_surface=0, friction_coeff_line=0)
mean_curvature = MeanCurvatureSolver(mesh, order=order, lset=levelset)
mean_curvature.Step()
fluid = TaylorHood(mesh, fluid_params, lset=levelset, nitsche_stab=100, f=CF((0, 0)), surface_tension=mean_curvature.H, dt=dt, 
                   order=order + 1, ghost_stab=1, add_convection=True, add_number_space=False, time_order=1, use_supg=False,
                  wall_params=wall_params, extension_radius=0.2)
fluid.SetOuterBoundaryCondition(NitscheVelocityBC(region="right|left", values=CF((0, 0))))
fluid.SetOuterBoundaryCondition(StrongNormalVelocityBC(region="bottom"))
fluid.Initialize()

sol = fluid.SolveStokes()
ngw.Draw(IfPos(levelset.lsetp1, CF((0, 0)), sol.components[0]), mesh)

# %%
velocity_extension = LevelsetBasedExtension(levelset, order=order, gamma=1e-1, ghost_stab=10, no_slip="top|left|right", no_penetration="bottom")

velocity_extension.SetRhs(fluid.gfu)
levelset.transport.SetWind(velocity_extension.field)

def should_finalize():
    return time_loop.i_inner == 3

end_time = 2

time_loop = TimeLoop(time=t, dt=dt, end_time=end_time, display_progress_bar=True, should_finalize=None)
time_loop.SetFinalizeRule(should_finalize)

cf_neg = Norm(fluid.gfu)
cf_pos = CF(0)
animation = UnfittedNGSWebguiPlot(levelset, cf_neg=cf_neg, cf_pos=cf_pos,
                                  order=fluid.order, time=t, end_time=end_time,
                                  name="animation", min=-0.075, max=0.4, autoscale=False)

time_loop.Register(velocity_extension, name="vel ext.")
time_loop.Register(levelset, name="levelset")
time_loop.Register(mean_curvature, name="mean curvature")
time_loop.Register(fluid, name="moving stokes")
time_loop.Register(animation, name="animation")

time_loop()
