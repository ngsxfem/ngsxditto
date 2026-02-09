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

kg = 1
dt = 0.1
h = 0.07
layers_proj = 1

mesh = Mesh(unit_square.GenerateMesh(maxh=h))
AdjaMat = AdjacencyMatrix(mesh, neighbortype="vertex")
t = Parameter(0)
wind = IfPos(t-2,1,-1)*CF((0.5-y, x-0.5))

# compare values
value_ds = 0
value_dx = 0

#%%

VGammaGF = GridFunction(H1(mesh, order=1), autoupdate=True) # to define delta for extension

# markers
transport_elems, old_transport_elems, support_elems = BitArray(mesh.ne), BitArray(mesh.ne), BitArray(mesh.ne)
transport_elems[:] = True
old_transport_elems[:] = True

# transport class
transport = ImplicitDGTransport(mesh, wind=wind, dt=dt, order=kg, active_elements=transport_elems)
levelset = LevelSetGeometry(transport)
fes_old_lset = transport.current.space
support_dofs = BitArray(fes_old_lset.ndof)
# lset_ext = ElementBasedExtension(gfs=[levelset.transport.past], supportelems=support_elems, targetelems=transport_elems)
lset_ext = ElementBasedExtension(gfs=[levelset.transport.past], supportelems=support_elems, targetelems=transport_elems, dirichlet_dofs=support_dofs)

# set initial values for the level set function
levelset.Initialize(sqrt((x-0.6)**2+(y-0.5)**2) - 0.15, initial_time = 0)
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
	support_dofs[:] = GetDofsOfElements(fes_old_lset, support_elems)

def Finish():
	global value_ds, value_dx

	# overwrite last element markers
	old_transport_elems[:] = transport_elems
	cont_vis.AddMultiDimComponent(levelset.field.vec)

	value_ds += sqrt(Integrate(CF(1) * levelset.dS, mesh))
	value_dx += sqrt(Integrate(levelset.field**2 * dx(definedonelements=transport_elems), mesh))


#%%
# create and run time loop
t.Set(0)
time_loop = TimeLoop(time=t, dt=dt, end_time=4)
time_loop.Register(UpdateElemMarker, name="udpate transport elements")
time_loop.Register(lset_ext, name="element based level set extension")
time_loop.Register(levelset, name="levelset")
time_loop.Register(Finish, name="Finish")
time_loop()

# %%
# final results
print(f"Difference to old code = {np.abs(value_ds - 9.710888279890247) + np.abs(value_dx - 0.9283146818444067)}")
Draw(cont_vis, mesh, height=f'{plot_height}px', width=f'{1.5*plot_height}px')

# %%
