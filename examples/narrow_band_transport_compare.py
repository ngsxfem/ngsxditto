from ngsolve import *
from ngsolve import unit_square, Mesh, CoefficientFunction, CF
from ngsolve.webgui import *

from ngsxditto.solver import *
from ngsxditto.levelset import *
from ngsxditto.extension import *
#from ngsxditto.transport import *

from xfem.utils import AdjacencyMatrix, AddNeighborhood
from utils_compare import CheckElementHistory, LevelSetTransport

# parameters and mesh
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

TaskManager().__enter__()
# inflow_values = CF(0)
transport_elems, old_transport_elems, support_elems = BitArray(mesh.ne), BitArray(mesh.ne), BitArray(mesh.ne)
transport_elems[:] = True
old_transport_elems[:] = True
# transport = ImplicitDGTransport(mesh, wind, inflow_values, dt, order=1, active_elements=transport_elems)
# levelset = LevelSetGeometry(transport)

# aux lset functions
H1lset = H1(mesh, order=kg, dgjumps=True)
H1_order1 = H1(mesh, order=1)
old_lset = GridFunction(H1lset)
VGammaGF = GridFunction(H1_order1, autoupdate=True) # to define delta for extension

# DOF marker
ProjectDofs = BitArray(H1lset.ndof)

# should be in levelset class
LsetTransport = LevelSetTransport(mesh=mesh, old_lset=old_lset, dt=dt, order=kg, active_elements=transport_elems, wind=wind)
levelset = LsetTransport.levelsetH1
LsetExtension = ElementBasedExtension(gfs=old_lset, supportelems=support_elems, targetelems=transport_elems, dirichlet_dofs=ProjectDofs)

ci_lset = CutInfo(mesh)
CutElem = ci_lset.GetElementsOfType(IF)
lsetmeshadap = LevelSetMeshAdaptation(mesh, order=kg, discontinuous_qn=False, heapsize=1000000000)
lsetp1 = lsetmeshadap.lset_p1
deformation = lsetmeshadap.deform
dGamma = dCut(levelset=lsetp1, domain_type=IF, definedonelements=CutElem, deformation=deformation, subdivlvl=None)
dxTrans = dx(definedonelements=transport_elems)

# set initial values for the level set function
# levelset.Initialize(sqrt((x-0.6)**2+(y-0.5)**2) - 0.15, initial_time = 0)
t.Set(0)
old_lset.Set(sqrt((x-0.6)**2+(y-0.5)**2) - 0.15)
levelset.vec.data = old_lset.vec # das hier muss drin bleiben, weil levelset ausßerhalb der transport_elems hiermit überschrieben wird
ci_lset.Update(levelset)
lsetmeshadap.CalcDeformation(levelset) # das updatet auch lsetp1



# Draw(transport.past, mesh, "udisc_past", min=-0.001, max=0.001, autoscale=False)
# Draw(transport.field, mesh, "udisc", min=-0.001, max=0.001, autoscale=False)
# Draw(levelset.field, mesh, "ucont", min=-0.001, max=0.001, autoscale=False)
# def redraw():
# 	Redraw(blocking=True)
# 	#input("")

def SymbolL2Norm(f, symbol, mesh) -> float:
	return sqrt(Integrate(InnerProduct(f, f).Compile() * symbol, mesh=mesh))

def UpdateElemMarker():
	# support_elems[:] = AddNeighborhood(levelset.hasif, AdjaMat, layers=1, inplace=False)
	# transport_elems[:] = AddNeighborhood(levelset.hasif, AdjaMat, layers=3, inplace=False)

	# calculate number of layers for transport step
	VGammaGF.Set(wind * Normalize(grad(levelset)), definedonelements = CutElem)
	max_VGamma = np.max(np.abs(VGammaGF.vec.FV().MinMax()))
	layers_trans = max(int(round(1 + 4*dt/h * max_VGamma)), 3)

	# update elemets where transport step is performed
	transport_elems[:] = AddNeighborhood(CutElem, AdjaMat, layers=layers_trans)
	support_elems[:] = AddNeighborhood(CutElem, AdjaMat, layers=layers_proj) & old_transport_elems & transport_elems

	ProjectDofs[:] = GetDofsOfElements(H1lset, support_elems)
	CheckElementHistory(support_elems, old_transport_elems) # ensures that old lsets are defined on support_elems, should always work, since we use the intersection with the old elements

def Finish():
	global value_ds, value_dx
	lsetmeshadap.CalcDeformation(levelset) # das updatet auch lsetp1
	ci_lset.Update(lsetp1)
	CheckElementHistory(CutElem, transport_elems)

	# overwrite last levelset and element markers
	old_lset.vec.data = levelset.vec
	old_transport_elems[:] = transport_elems

	value_ds += SymbolL2Norm(CF(1), dGamma, mesh)
	value_dx += SymbolL2Norm(levelset, dxTrans, mesh)

# create time loop
time_loop = TimeLoop(time=t, dt=dt, end_time=1)
time_loop.Register(UpdateElemMarker, name="udpate transport elements")
time_loop.Register(LsetExtension, name="element based level set extension")
time_loop.Register(LsetTransport.Step, name="levelset")
time_loop.Register(Finish, name="Finish")


# %%
# run time loop
time_loop()

print(f"Difference to old code = {np.abs(value_ds - 9.720120260330926) + np.abs(value_dx - 0.9684255662101534)}")



# ebext = ElementBasedExtension(levelset.transport.past, support_elems, transport_elems)

#input("")
# create time loop
# time_loop = TimeLoop(time=t, dt=dt, end_time=1)
# time_loop.Register(UpdateElemMarker, name="udpate transport elements")
# time_loop.Register(ebext, name="element based level set extension")
# time_loop.Register(levelset, name="levelset")
# time_loop.Register(redraw, name="redraw")


# # %%
# # run time loop
# time_loop()
