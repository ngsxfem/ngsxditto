#%%
from ngsolve import * 

from ngsolve import unit_square, Mesh, CoefficientFunction, CF

from ngsolve.webgui import *

from time import sleep
#%%
mesh = Mesh(unit_square.GenerateMesh(maxh=0.02))

#%%
from ngsxditto.extension import *
from xfem.utils import AdjacencyMatrix, AddNeighborhood
V = H1(mesh,order=5, dgjumps=True, dirichlet=".*")
gfu = GridFunction(V)
gfw = GridFunction(V)


lsetp1 = GridFunction(H1(mesh,order=1))
lsetp1.Set(sqrt((x-0.5)**2 + (y-0.5)**2) - 0.3)

ci = CutInfo(mesh, lsetp1)
negels = ci.GetElementsOfType(NEG)
ifels = ci.GetElementsOfType(IF)
hasneg = ci.GetElementsOfType(HASNEG)
haspos = ci.GetElementsOfType(HASPOS)
anyelems = ci.GetElementsOfType(ANY)
#%%

#band = BitArray(mesh.ne)
band = AddNeighborhood(ifels, AdjacencyMatrix(mesh,"vertex"), layers=3)
#ifels = ci.GetElementsOfType(IF)


theta = atan2(y-0.5,x-0.5)
gfu.Set(sin(theta), definedonelements=ifels)

#gfu.Set(sin(4*pi*x*y) , definedonelements=ifels)

u,v = V.TnT()
n = Normalize(grad(lsetp1))
energyform = (grad(u)|n)*(grad(v)|n)*dx(definedonelements=band)

oldband = ifels

ebext = ElementBasedExtension(gfu, oldband, band, 
                              energyform=energyform,
                              activeelems=band,
                              activefacets=None
                            )

                              #dirichlet_dofs=~V.FreeDofs())
#%%
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
