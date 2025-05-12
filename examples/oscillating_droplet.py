#%%
from ngsolve import * 

from ngsolve import unit_square, Mesh, CoefficientFunction, CF

from time import sleep
#%%
mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))

#%%
from ngsxditto.transport import *
from ngsxditto.meancurv import *


t = 0
tend = 1
dt = 0.01

fluid_params = FluidParameters(nu=1e-3) # dictionary of default fluid parameters + changes from arguments

fluid = H1ConformingFluid(mesh, order, fluid_params, levelset=None) # inherited from FluidDiscretization

transport = ExplicitDGTransport(mesh, dt=0.01)

redistancing = FastMarching(transport.field) # inherited from Redistancing

levelset = LevelSetGeometry(transport, redistancing) # holds LevelSetMeshAdaptation & CutInfo 

transport.SetWind(fluid.velocity, inflow_values=fluid.velocity)

levelset.Initialize( initial_lset )
fluid.SetLevelSet( levelset )
fluid.Initialize( initial_velocity ) # or: fluid.SolveStokes( bnd_data, forces )

while t < tend:
    fluid.DoOneStep(..., dt)
    levelset.transport.DoOneStep(...)
    t += dt