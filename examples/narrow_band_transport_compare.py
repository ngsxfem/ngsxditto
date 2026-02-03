#%%
# imports
from ngsxditto.solver import *
from ngsxditto.levelset import *
from ngsxditto.extension import *
# from ngsxditto.transport import *

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
# aux lset functions
H1lset = H1(mesh, order=kg, dgjumps=True)
old_lset = GridFunction(H1lset)
VGammaGF = GridFunction(H1(mesh, order=1), autoupdate=True) # to define delta for extension

# markers
transport_elems, old_transport_elems, support_elems = BitArray(mesh.ne), BitArray(mesh.ne), BitArray(mesh.ne)
transport_elems[:] = True
old_transport_elems[:] = True
support_dofs = BitArray(H1lset.ndof)

# transport class
transport = ImplicitDGTransport(mesh, wind=wind, inflow_values=old_lset, dt=dt, order=kg, active_elements=transport_elems)
levelset = LevelSetGeometry(transport)
lset_ext = ElementBasedExtension(gfs=[old_lset, levelset.transport.past], supportelems=support_elems, targetelems=transport_elems, dirichlet_dofs=support_dofs)

# set initial values for the level set function
levelset.Initialize(sqrt((x-0.6)**2+(y-0.5)**2) - 0.15, initial_time = 0)
t.Set(0)
old_lset.Set(sqrt((x-0.6)**2+(y-0.5)**2) - 0.15)

#%%
# visualization
cont_vis = GridFunction(levelset.field.space, multidim=0)
oldcont_vis = GridFunction(old_lset.space, multidim=0)
olddisc_vis = GridFunction(levelset.transport.past.space, multidim=0)
olddiff_vis = GridFunction(levelset.field.space, multidim=0)
diff_gf = GridFunction(levelset.field.space)

# Draw(transport.past, mesh, "udisc_past", height=f'{plot_height}px', width=f'{1.5*plot_height}px')
# Draw(old_lset, mesh, "ucont_past", height=f'{plot_height}px', width=f'{1.5*plot_height}px')
# Draw(transport.field, mesh, "udisc", height=f'{plot_height}px', width=f'{1.5*plot_height}px')
# Draw(levelset.field, mesh, "ucont", height=f'{plot_height}px', width=f'{1.5*plot_height}px')

# Draw(transport.past, mesh, "udisc_past", min=-0.001, max=0.001, autoscale=False, height=f'{plot_height}px', width=f'{1.5*plot_height}px')
# Draw(old_lset, mesh, "ucont_past", min=-0.001, max=0.001, autoscale=False, height=f'{plot_height}px', width=f'{1.5*plot_height}px')
# Draw(transport.field, mesh, "udisc", min=-0.001, max=0.001, autoscale=False, height=f'{plot_height}px', width=f'{1.5*plot_height}px')
# Draw(levelset.field, mesh, "ucont", min=-0.001, max=0.001, autoscale=False, height=f'{plot_height}px', width=f'{1.5*plot_height}px')

def SymbolL2Norm(f, symbol, mesh) -> float:
	return sqrt(Integrate(InnerProduct(f, f).Compile() * symbol, mesh=mesh))

def UpdateElemMarker():
	# calculate number of layers for transport step
	VGammaGF.Set(wind * levelset.n, definedonelements = levelset.hasif)
	max_VGamma = np.max(np.abs(VGammaGF.vec.FV().MinMax()))
	layers_trans = max(int(round(1 + 4*dt/h * max_VGamma)), 3)

	# update element and dof markers
	transport_elems[:] = AddNeighborhood(levelset.hasif, AdjaMat, layers=layers_trans)
	support_elems[:] = AddNeighborhood(levelset.hasif, AdjaMat, layers=layers_proj) & old_transport_elems & transport_elems
	support_dofs[:] = GetDofsOfElements(H1lset, support_elems)

def SaveToDraw():
	cont_vis.AddMultiDimComponent(levelset.field.vec)
	oldcont_vis.AddMultiDimComponent(old_lset.vec)
	olddisc_vis.AddMultiDimComponent(levelset.transport.past.vec)
	diff_gf.Set(levelset.transport.past - old_lset, definedonelements=transport_elems)
	olddiff_vis.AddMultiDimComponent(diff_gf.vec)

def Finish():
	global value_ds, value_dx

	# overwrite last levelset and element markers
	old_lset.vec.data = levelset.field.vec
	old_transport_elems[:] = transport_elems

	value_ds += SymbolL2Norm(CF(1), levelset.dS, mesh)
	value_dx += SymbolL2Norm(levelset.field, dx(definedonelements=transport_elems), mesh)


#%%
# create and run time loop
t.Set(0)
time_loop = TimeLoop(time=t, dt=dt, end_time=1)
time_loop.Register(UpdateElemMarker, name="udpate transport elements")
time_loop.Register(lset_ext, name="element based level set extension")
time_loop.Register(SaveToDraw, name="for drawing")
time_loop.Register(levelset, name="levelset")
time_loop.Register(Finish, name="Finish")
time_loop()

# %%
# final results
print(f"Difference to old code = {np.abs(value_ds - 9.720120260330926) + np.abs(value_dx - 0.9684255662101534)}")
# Draw(cont_vis, mesh, height=f'{plot_height}px', width=f'{1.5*plot_height}px')
Draw(oldcont_vis, mesh, height=f'{plot_height}px', width=f'{1.5*plot_height}px')
Draw(olddiff_vis, mesh, height=f'{plot_height}px', width=f'{1.5*plot_height}px')


# %%
