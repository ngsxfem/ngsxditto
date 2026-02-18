#%%
# imports
from ngsxditto.solver import *
from ngsxditto.levelset import *
from ngsxditto.extension import *

from xfem.utils import AdjacencyMatrix, AddNeighborhood
from ngsolve.webgui import *

#%%
# parameters and mesh
order = 3
dt = 0.05
h = 0.07
layers_proj = 1

mesh = Mesh(unit_square.GenerateMesh(maxh=h))
AdjaMat = AdjacencyMatrix(mesh, neighbortype="vertex")
t = Parameter(0)
wind = IfPos(t-2,1,-1)*CF((0.5-y, x-0.5))

#%%
# aux gridfunction
VGammaGF = GridFunction(H1(mesh, order=1)) # to define bandwidth for narrow band

# element markers
transport_elems, support_elems = BitArray(mesh.ne), BitArray(mesh.ne)
transport_elems[:] = True

# transport class
transport = ImplicitDGTransport(mesh, wind=wind, dt=dt, order=order, active_elements=transport_elems)
levelset = LevelSetGeometry(transport)

ext_support_dofs = BitArray(levelset.transport.past_cont.space.ndof)
lset_ext = ElementBasedExtension(levelset.transport.past_cont, supportelems=support_elems, targetelems=transport_elems, dirichlet_dofs=ext_support_dofs)

# set initial values for the level set function
levelset.Initialize(sqrt((x-0.6)**2+(y-0.5)**2) - 0.15, initial_time=0)

#%%
# visualization
vis = GridFunction(levelset.field.space, multidim=0)

def UpdateElemMarker():
	# calculate number of layers for transport step
	VGammaGF.Set(wind * levelset.n, definedonelements = levelset.hasif)
	max_VGamma = np.max(np.abs(VGammaGF.vec.FV().MinMax()))
	layers_trans = max(int(round(1 + 4*dt/h * max_VGamma)), 3)

	# update element and dof markers
	support_elems[:] = AddNeighborhood(levelset.hasif, AdjaMat, layers=layers_proj) & transport_elems	# cut with old transport elems because levelset_past is defined there
	transport_elems[:] = AddNeighborhood(levelset.hasif, AdjaMat, layers=layers_trans)
	ext_support_dofs[:] = GetDofsOfElements(levelset.transport.past_cont.space, support_elems)

def VisualizeStep():
	vis.AddMultiDimComponent(levelset.field.vec)

#%%
# create and run time loop
time_loop = TimeLoop(time=t, dt=dt, end_time=4)
time_loop.Register(UpdateElemMarker, name="udpate transport elements")
time_loop.Register(lset_ext, name="element based level set extension")
time_loop.Register(levelset, name="levelset")
time_loop.Register(VisualizeStep, name="Save for visualize")
time_loop()

# %%
Draw(vis, mesh, height='600px', width='900px')

# %%
