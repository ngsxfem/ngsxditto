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

# %% [markdown]
# # An unfitted time-dependant Stokes problem

# %% [markdown]
# We can solve the Stokes problem on a moving (unfitted) domain. Here we show an example of a rising ball that the fluid moves around.

# %% jupyter={"outputs_hidden": true}
from ngsolve import *
from xfem import *
import ngsolve.webgui as ngw
from netgen.occ import *

from ngsxditto.transport import *
from ngsxditto.levelset import *
from ngsxditto.fluid import *
from ngsxditto.redistancing import *
from ngsxditto.visualization import *
from ngsxditto.solver import *

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
starting_levelset = -((x**2 + (y+0.5-t)**2)**(1/2) - 0.5)
transport = KnownSolutionTransport(mesh, starting_levelset, dt=dt, order=order)
levelset = LevelSetGeometry(transport)
levelset.Initialize(starting_levelset)
ngw.Draw(levelset.field)
# %% [markdown]
# We define the fluid and solve the stationary Stokes problem for the starting levelset position.

# %%
fluid_params = FluidParameters(viscosity=1e-1)
fluid = TaylorHood(mesh, fluid_params, order=order, lset=levelset, dt=dt)

fluid.SetOuterBoundaryCondition(StrongDirichletBC(region="left", values=CF(((1-y)*(1+y), 0))))
fluid.SetOuterBoundaryCondition(StrongDirichletBC(region="top|bottom", values=CF((0, 0))))
fluid.SetInnerBoundaryCondition(CF((0, 0)))

fluid.Initialize()
sol = fluid.SolveStokes()
gfu, gfp = sol.components
fluid.SetInitialValues(gfu, gfp)
ngw.Draw(IfPos(levelset.field, CF((0, 0)), fluid.gfu), mesh)
ngw.Draw(IfPos(levelset.field, CF(0), fluid.gfp), mesh)
# %%
end_time = 1

time_loop = TimeLoop(time=t, dt=dt, end_time=end_time)

cf_neg = Norm(fluid.gfu)
cf_pos = -1
animation = UnfittedNGSWebguiPlot(levelset, cf_neg=cf_neg, cf_pos=cf_pos,
                                  order=fluid.order, time=t, end_time=end_time,
                                  name="animation", min=0, max=2, autoscale=False)


time_loop.Register(levelset, name="levelset update")
time_loop.Register(fluid, name="fluid update")
time_loop.Register(animation, name="animation")

time_loop()
