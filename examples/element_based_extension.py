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
mesh = Mesh(unit_square.GenerateMesh(maxh=0.03))

# from netgen.geom2d import SplineGeometry
# background_domain = SplineGeometry()
# background_domain.AddRectangle((-2, -2), (2, 2))
# mesh = Mesh(background_domain.GenerateMesh(maxh=0.1, quad_dominated=False))

V = H1(mesh, order=5, dgjumps=True, dirichlet=".*")
gfu = GridFunction(V)

exact_levelset = 2*(((x-0.55) + 3*(y-0.5)**2)**2 + (y-0.5)**2 - 0.07)

lsetp1 = GridFunction(H1(mesh,order=1))
# lsetp1.Set(sqrt((x-0.5)**2 + (y-0.5)**2) - 0.3)
lsetp1.Set(exact_levelset)
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
# oldband = ifels
oldband = AddNeighborhood(ifels, AdjacencyMatrix(mesh,"vertex"), layers=1)
band = AddNeighborhood(oldband, AdjacencyMatrix(mesh,"vertex"), layers=1)

# input function defined on first oldband
theta = atan2(y-0.5,x-0.5)
gfu.Set(sin(theta), definedonelements=oldband)
gfu.Set(exact_levelset, definedonelements=oldband)

# choose normal volume energy form for extension
# u, v = V.TnT()
# n = Normalize(grad(lsetp1))
# energyform = (grad(u)|n)*(grad(v)|n)*dx(definedonelements=band)

ebext = ElementBasedExtension(gfu, oldband, band,
                              # energyform=energyform,
                              # activeelems=band,
                              # activefacets=None
                            )
# # alternatively, use default energy form (ghost penalty)
# ebext = ElementBasedExtension(gfu, oldband, band)


#%%
# store output in multidim gridfunction
gfu_plot = GridFunction(V)
InactiveDofs = ~GetDofsOfElements(V, band)
gfu_plot.vec.data = gfu.vec
gfu_plot.vec[InactiveDofs] = np.nan
vtk = VTKOutput(mesh, coefs=[gfu_plot], names=["solution"], filename="animation/animation_data", subdivision=0)

#%%
# vtk.Do()

gfwvis = GridFunction(V, multidim=0)
gfwvis.AddMultiDimComponent(gfu.vec)
ebext.Step()
InactiveDofs = ~GetDofsOfElements(V, band)
gfu_plot.vec.data = gfu.vec
gfu_plot.vec[InactiveDofs] = np.nan

vtk.Do()
gfwvis.AddMultiDimComponent(gfu.vec)


for i in range(8):
    oldband[:] = band
    AddNeighborhood(band, AdjacencyMatrix(mesh,"vertex"), layers=1, inplace=True)
    ebext.Step()
    InactiveDofs = ~GetDofsOfElements(V, band)
    gfu_plot.vec.data = gfu.vec
    gfu_plot.vec[InactiveDofs] = np.nan
    vtk.Do()
    gfwvis.AddMultiDimComponent(gfu.vec)
Draw(gfwvis, mesh, "gfwvis")
# %%
