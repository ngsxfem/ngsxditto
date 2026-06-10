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

# %%
from ngsxditto.utils.loglevel import loggingSlider
loggingSlider(default_level="WARNING") # global level
loggingSlider("ngsxditto", default_level="DEBUG") # ngsxditto

# %% [markdown]
# # Fluid movement in a glass

# %% [markdown]
# We consider a "glass" filled with a fluid. Starting with a uneven surface the gravity and surface tension forces lead to a swaying movent of the fluid.

# %% [markdown]
# Firstly, we create the domain and the levelset that describes the surface of the fluid.

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
mesh = Mesh(OCCGeometry(domain, dim=2).GenerateMesh(maxh=0.1))

# %%
dt = 4e-2
order = 1
t = Parameter(0)
starting_levelset = y - 0.3*x
transport = ExplicitDGTransport(mesh, dt=dt, order=order, compile=False)
levelset = LevelSetGeometry(transport)
levelset.Initialize(starting_levelset)
ngw.Draw(levelset.field)

# %% [markdown]
# We now create the fluid object. We assume no normal flow at the left and right side $u \cdot n = 0$ and no velocity at the bottom boundary. For the surface we assume free boundary conditions.

# %%
fluid_params = FluidParameters(viscosity=1e-2, surface_tension_coeff=1e-2)

mean_curvature = MeanCurvatureSolver(mesh, order=order, lset=levelset)
mean_curvature.Step()
fluid = TaylorHood(mesh, fluid_params, lset=levelset, nitsche_stab=100, f=CF((0, -9.8)), surface_tension=mean_curvature.H, dt=dt, order=order + 1, ghost_stab=1,
                   add_convection=True, add_number_space=False, extension_radius=0.3, time_order=1, use_supg=True)
fluid.SetOuterBoundaryCondition(NitscheNormalVelocityBC(region="right|left", values=CF(0)))
fluid.SetOuterBoundaryCondition(StrongDirichletBC(region="bottom", values=CF((0, 0))))
fluid.Initialize()

sol = fluid.SolveStokes()
ngw.Draw(IfPos(levelset.lsetp1, CF((0, 0)), sol.components[0]), mesh)

# %% [markdown]
# As for the oscillating droplet example we use a velocity extension to move the levelset according to the fluid velocity.

# %%
velocity_extension = LevelsetBasedExtension(levelset, order=order, gamma=1e-1, ghost_stab=1, no_slip="top|bottom")

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
                                  name="animation", min=-0.075, max=0.5, autoscale=False)

time_loop.Register(velocity_extension, name="vel ext.")
time_loop.Register(levelset, name="levelset")
time_loop.Register(mean_curvature, name="mean curvature")
time_loop.Register(fluid, name="moving stokes")
time_loop.Register(animation, name="animation")

# %%
time_loop()

# %%
