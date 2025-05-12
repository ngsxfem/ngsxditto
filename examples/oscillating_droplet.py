#%%
from ngsolve import * 

from ngsolve import unit_square, Mesh, CoefficientFunction, CF

from time import sleep
#%%
mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))

#%%
from ngsxditto.transport import *
from ngsxditto.levelset import *
from ngsxditto.redistancing import *
from ngsxditto.fluid import *


t = 0
tend = 1
dt = 0.01

fluid_params = FluidParameters(viscosity=1e-3) # dictionary of default fluid parameters + changes from arguments (dynamic_viscosity, kinematic_viscosity, density, surface_tension_coeff)
wall_params = WallParameters(friction_coeff=..., contact_angle=...)

fluid = H1ConformingFluid(mesh, order, fluid_params, levelset=None, wall_params=None) # inherited from FluidDiscretization

transport = ExplicitDGTransport(mesh, dt=0.01) # different other options...

redistancing = FastMarching(transport.field) # inherited from Redistancing / can be None

levelset = LevelSetGeometry(transport, redistancing) # holds LevelSetMeshAdaptation & CutInfo 

transport.SetWind(fluid.velocity, inflow_values=fluid.velocity)

levelset.Initialize( initial_lset )
fluid.SetLevelSet( levelset )
fluid.Initialize( initial_velocity ) # or: fluid.SolveStokes( bnd_data, forces )

while t < tend:
    fluid.DoOneStep(..., dt)
    levelset.transport.DoOneStep(...)
    t += dt