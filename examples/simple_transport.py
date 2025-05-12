#%%
from ngsolve import * 

from ngsolve import unit_square, Mesh, CoefficientFunction, CF

from time import sleep
#%%
mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))

#%%
from ngsxditto.transport import *

wind = CF((0.5-y, x-0.5))
inflow_values = CF(0)
transport = ExplicitDGTransport(mesh, wind, inflow_values, 0.01)

#%%
transport.SetInitialValues(exp((-4*((x-0.6)**2+(y-0.5)**2))), initial_time = 0)

#%%
Draw(transport.field, mesh, "u")

def redraw_callback():
    sleep(0.01)
    Redraw()

transport.AddCallBack(redraw_callback)

transport.Propagate(0, 1)

# %%
