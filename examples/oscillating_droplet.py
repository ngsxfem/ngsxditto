# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
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
# # Example: Oscillating Droplet

# %% [markdown]
# In this example we start with an elliptic droplet. The surface tension force deforms it as it strives to minimize the surface. In other words it will always try to form the droplet towards a cirlce. Due to the viscosity however it will slightly overcorrect such that a wobbling movement occurs. The following PDE describes this problem:
# $$
# \begin{align*}
# \rho \partial_t \mathbf{u}
#   - \operatorname{div}(2 \mu \varepsilon(\mathbf{u})) + \nabla p &= 0
#   && \text{in } {\Omega(t)} \\
# \nabla \cdot \mathbf{u} &= 0
#   && \text{in } {\Omega(t)} \\
# \boldsymbol{\sigma} \cdot \mathbf{n}_\Gamma &= {\tau \kappa \mathbf{n}_\Gamma}
#   && \text{on } {\Gamma(t)} \\
#   { \mathbf{u} \cdot \mathbf{n}_\Gamma} &{= \mathcal{V}_\Gamma} && \text{on } {\Gamma(t)}
# \end{align*}
# $$
# where $\tau$ is a surface tension coefficient, $\kappa$ is the mean curvature and $\mathcal{V}_\Gamma$ is the velocity of the interface in normal direction.
# %% [markdown]
# First we import the necessary modules and create the mesh

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

# %% [markdown]
# We define our initial geometry.

# %%
dt = 0.05
order = 2
t = Parameter(0)
starting_levelset = (5*x**2 + y**2)**(1/2) - 2.0/3.0
transport = ExplicitDGTransport(mesh, dt=dt, order=order, compile=False)
levelset = LevelSetGeometry(transport)
levelset.Initialize(starting_levelset)
ngw.Draw(levelset.field)
# %% [markdown]
# We use Taylor-Hood elements. As a simplification we only consider the (unsteady) Stokes problem. We calculate the curvature of the level set geometry and add the resulting tension force to the right hand side in our Stokes solver. We assume we have no initial velocity.

# %%
fluid_params = FluidParameters(viscosity=1e-2)

mean_curvature = MeanCurvatureSolver(mesh, order=order, lset=levelset)
mean_curvature.Step()

fluid = TaylorHood(mesh, fluid_params, lset=levelset, f=CF((0, 0)), surface_tension=mean_curvature.H, dt=dt, order=order + 1, ghost_stab=1)
fluid.Initialize(initial_velocity=CF((0, 0)))
# %% [markdown]
# For the unsteady Stokes problem we now want to update our level set based on this field, i.e. $$\mathbf{u} \cdot \mathbf{n}_\Gamma = \mathcal{V}_\Gamma$$ where $\mathcal{V}_\Gamma$ is the velocity of the interface in normal direction. For our level set update we need a velocity field $w$ on the whole domain, not just on the interface. For this we extend the velocity field using a diffusion based algorithm. After our level set update we can then calculate the curvature again to solve a time-step of the Stokes problem.

# %%
velocity_extension = LevelsetBasedExtension(levelset, gamma=1e-3, order=order)
velocity_extension.SetRhs(fluid.gfu)
levelset.transport.SetWind(velocity_extension.field)

end_time = 2

time_loop = TimeLoop(time=t, dt=dt, end_time=end_time)

sphericity = SphericityDiagram(levelset, time=t, name="sphericity")

cf_neg = Norm(fluid.gfu)
cf_pos = -1
animation = UnfittedNGSWebguiPlot(levelset, cf_neg=cf_neg, cf_pos=cf_pos,
                                  order=fluid.order, time=t, end_time=end_time,
                                  name="animation", min=-0.075, max=0.225, autoscale=False)

pv_vis = PyVistaAnimation(mesh=mesh, lset=levelset, cf_neg=cf_neg,
                           subdivision=3)

time_loop.Register(velocity_extension, name="vel ext.")
time_loop.Register(levelset, name="levelset")
time_loop.Register(mean_curvature, name="mean curvature")
time_loop.Register(fluid, name="moving stokes")

time_loop.Register(sphericity, name="sphericity")
time_loop.Register(animation, name="animation")
time_loop.Register(pv_vis, name="pyvista")

time_loop()

# %% [markdown]
# The sphericity diagram shows the ratio of the surface area (perimeter) and the volume (area) of a shape. We observe that starting from an initial deformation this ratio has periodic declining behavior.
