#%%
# imports
from ngsolve import *

from ngsolve import unit_square, Mesh, CoefficientFunction, CF

from ngsolve.webgui import *

from ngsxditto.extension import *

from xfem.utils import AdjacencyMatrix, AddNeighborhood

from time import sleep
#%%
# mesh and gridfunctions
mesh = Mesh(unit_square.GenerateMesh(maxh=0.02))
V = H1(mesh, order=5, dgjumps=True, dirichlet=".*")
gfu = GridFunction(V)

lsetp1 = GridFunction(H1(mesh,order=1))
lsetp1.Set(sqrt((x-0.5)**2 + (y-0.5)**2) - 0.3)
ci = CutInfo(mesh, lsetp1)

#%%
# element markers
negels = ci.GetElementsOfType(NEG)
ifels = ci.GetElementsOfType(IF)
hasneg = ci.GetElementsOfType(HASNEG)
haspos = ci.GetElementsOfType(HASPOS)
anyelems = ci.GetElementsOfType(ANY)

#%%
# oldband = elems where function is defined previously, band = elems where function is defined after extension
oldband = ifels
band = AddNeighborhood(ifels, AdjacencyMatrix(mesh,"vertex"), layers=3)

# input function defined on first oldband
theta = atan2(y-0.5,x-0.5)
gfu.Set(sin(theta), definedonelements=oldband)
#gfu.Set(sin(4*pi*x*y) , definedonelements=oldband)

# choose normal volume energy form for extension
u, v = V.TnT()
n = Normalize(grad(lsetp1))
energyform = (grad(u)|n)*(grad(v)|n)*dx(definedonelements=band)

ebext = ElementBasedExtension(gfu, oldband, band,
                              energyform=energyform,
                              activeelems=band,
                              activefacets=None
                            )
# # alternatively, use default energy form (ghost penalty)
# ebext = ElementBasedExtension(gfu, oldband, band)


#%%
# store output in multidim gridfunction
gfwvis = GridFunction(V, multidim=0)
gfwvis.AddMultiDimComponent(gfu.vec)
ebext.Step()

for i in range(20):
    oldband[:] = band
    AddNeighborhood(band, AdjacencyMatrix(mesh,"vertex"), layers=1, inplace=True)
    ebext.Step()
    gfwvis.AddMultiDimComponent(gfu.vec)
Draw(gfwvis, mesh, "gfwvis")
# %%
