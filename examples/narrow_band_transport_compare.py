#%%
# imports
from ngsxditto.solver import *
from ngsxditto.levelset import *
from ngsxditto.extension import *

from xfem.utils import AdjacencyMatrix, AddNeighborhood
from ngsolve.webgui import *

#%%
# parameters and mesh
plot_height = 600

kg = 3
dt = 0.05
h = 0.07
layers_proj = 1

mesh = Mesh(unit_square.GenerateMesh(maxh=h))
AdjaMat = AdjacencyMatrix(mesh, neighbortype="vertex")
t = Parameter(0)
wind = IfPos(t-2,1,-1)*CF((0.5-y, x-0.5))

#%%
# aux gridfunctions
VGammaGF = GridFunction(H1(mesh, order=1), autoupdate=True) # to define delta for extension
transport_past_cont = GridFunction(H1(mesh, order=kg, dgjumps=True))

# markers
transport_elems, old_transport_elems, support_elems = BitArray(mesh.ne), BitArray(mesh.ne), BitArray(mesh.ne)
transport_elems[:] = True
old_transport_elems[:] = True
support_dofs = BitArray(transport_past_cont.space.ndof)

# transport class
transport = ImplicitDGTransport(mesh, wind=wind, dt=dt, order=kg, active_elements=transport_elems)
levelset = LevelSetGeometry(transport)
lset_ext = ElementBasedExtension(gfs=[transport_past_cont], supportelems=support_elems, targetelems=transport_elems, dirichlet_dofs=support_dofs)
# lset_ext = ElementBasedExtension(gfs=[levelset.transport.past], supportelems=support_elems, targetelems=transport_elems)

# set initial values for the level set function
levelset.Initialize(sqrt((x-0.6)**2+(y-0.5)**2) - 0.15, initial_time = 0)
transport_past_cont.Set(sqrt((x-0.6)**2+(y-0.5)**2) - 0.15)
t.Set(0)

#%%
# visualization
cont_vis = GridFunction(levelset.field.space, multidim=0)

def UpdateElemMarker():
	# calculate number of layers for transport step
	VGammaGF.Set(wind * levelset.n, definedonelements = levelset.hasif)
	max_VGamma = np.max(np.abs(VGammaGF.vec.FV().MinMax()))
	layers_trans = max(int(round(1 + 4*dt/h * max_VGamma)), 3)

	# update element and dof markers
	transport_elems[:] = AddNeighborhood(levelset.hasif, AdjaMat, layers=layers_trans)
	support_elems[:] = AddNeighborhood(levelset.hasif, AdjaMat, layers=layers_proj) & old_transport_elems & transport_elems
	support_dofs[:] = GetDofsOfElements(transport_past_cont.space, support_elems)

def InterExt():
	levelset.transport.past.Set(transport_past_cont)

def Finish():
	# overwrite last element markers
	old_transport_elems[:] = transport_elems
	transport_past_cont.Set(levelset.field)
	cont_vis.AddMultiDimComponent(levelset.field.vec)


#%%
# create and run time loop
t.Set(0)
time_loop = TimeLoop(time=t, dt=dt, end_time=4)
time_loop.Register(UpdateElemMarker, name="udpate transport elements")
time_loop.Register(lset_ext, name="element based level set extension")
time_loop.Register(InterExt, name="set old level set")
time_loop.Register(levelset, name="levelset")
time_loop.Register(Finish, name="Finish")
time_loop()

# %%
Draw(cont_vis, mesh, height=f'{plot_height}px', width=f'{1.5*plot_height}px')

# %%
