#%%
from ngsolve import *

from ngsolve import unit_square, Mesh, CoefficientFunction, CF
from ngsolve.webgui import *

from ngsxditto.levelset import *
# from ngsxditto.solver import *
# from ngsxditto.extension import *
#from ngsxditto.transport import *
from xfem.utils import AdjacencyMatrix, AddNeighborhood

# from time import sleep
#%%
# mesh and parameters
mesh = Mesh(unit_square.GenerateMesh(maxh=0.07))
dt = 0.01
t = Parameter(0)
adj = AdjacencyMatrix(mesh,"vertex")

#%%
wind = IfPos(t-2,1,-1)*CF((0.5-y, x-0.5))
#wind = CF((-1,0))
inflow_values = CF(0)
transport_elems = BitArray(mesh.ne)
transport_elems[:] = True
support_elems = BitArray(mesh.ne)
transport = ImplicitDGTransport(mesh, wind, inflow_values, dt,
								order=1, active_elements=transport_elems)
levelset = LevelSetGeometry(transport)
#%%
# set initial values for the level set function
#levelset.Initialize(exp((-4*((x-0.6)**2+(y-0.5)**2)))-0.8, initial_time = 0)
levelset.Initialize(sqrt((x-0.6)**2+(y-0.5)**2) - 0.15, initial_time = 0)
#%%
Draw(transport.past, mesh, "udisc_past", min=-0.001, max=0.001, autoscale=False)
Draw(transport.field, mesh, "udisc", min=-0.001, max=0.001, autoscale=False)
Draw(levelset.field, mesh, "ucont", min=-0.001, max=0.001, autoscale=False)

#%%
def UpdateElemMarker():
	support_elems[:] = AddNeighborhood(levelset.hasif, adj, layers=1, inplace=False)
	# support_elems &= transport_elems
	transport_elems[:] = AddNeighborhood(levelset.hasif, adj, layers=3, inplace=False)

target_elems = BitArray(mesh.ne)
target_elems[:] = True

ebext = ElementBasedExtension(levelset.transport.past, support_elems, transport_elems)
#ebext = ElementBasedExtension(levelset.transport.past, support_elems, target_elems)

#input("")

#%%
# create time loop

def redraw():
	Redraw(blocking=True)
	#input("")

time_loop = TimeLoop(time=t, dt=dt, end_time=4)
time_loop.Register(UpdateElemMarker, name="udpate transport elements")
time_loop.Register(ebext, name="element based level set extension")
time_loop.Register(levelset, name="levelset")
time_loop.Register(redraw, name="redraw")


# %%
# run time loop
time_loop()
# %%
