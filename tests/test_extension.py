from ngsolve import *
from ngsxditto import *
from ngsxditto.new_xfem_functions import *



domain = MoveTo(-1, -1).Rectangle(2, 2).Face()
domain.edges.Max(X).name = "right"
domain.edges.Min(X).name = "left"
domain.edges.Min(Y).name = "bottom"
domain.edges.Max(Y).name = "top"
mesh = Mesh(OCCGeometry(domain, dim=2).GenerateMesh(maxh=0.1))
levelset_function = CF(y)
levelset = LevelSetGeometry.from_cf(levelset_function, mesh)


def test_levelset_based():
    u_exact = CF((0, 1 + x ** 2))
    lset_ext = LevelsetBasedExtension(levelset, dirichlet="")

    lset_ext.SetRhs(u_exact)
    lset_ext.Step()

    l2_error = Integrate((u_exact - lset_ext.field)**2 * dx, mesh)**(1/2)
    assert l2_error < 1e-9


def test_element_based_ghost_penalty():
    levelset_function = CF(y)
    levelset = LevelSetGeometry.from_cf(levelset_function, mesh)
    u_exact = CF(1 + x ** 2)
    ifels = levelset.hasif
    V = H1(mesh, order=2, dgjumps=True, dirichlet=".*")
    gfu = GridFunction(V)

    gfu.Set(u_exact, definedonelements=ifels)
    band = AddNeighborhood(ifels, AdjacencyMatrix(mesh, "vertex"), layers=1)
    oldband = ifels

    ebext = ElementBasedExtension(gfu, oldband, band, activeelems=band, activefacets=None)
    ebext.Step()

    for i in range(9):
        oldband[:] = band
        AddNeighborhood(band, AdjacencyMatrix(mesh, "vertex"), layers=1, inplace=True)
        ebext.Step()
        assert Integrate((u_exact - gfu)**2 * dx(definedonelements=band), mesh) ** (1 / 2) < 1e-10

        # region gets larger
        assert Integrate((BitArrayCF(band) - BitArrayCF(oldband)) * dx, mesh) > 0


def test_element_based_diffusion():
    levelset_function = CF(y)
    levelset = LevelSetGeometry.from_cf(levelset_function, mesh)
    u_exact = CF(1 + x**2)
    ifels = levelset.hasif
    V = H1(mesh, order=2, dgjumps=True, dirichlet=".*")
    gfu = GridFunction(V)

    gfu.Set(u_exact, definedonelements=ifels)
    band = AddNeighborhood(ifels, AdjacencyMatrix(mesh, "vertex"), layers=1)

    u, v = V.TnT()
    energyform = (grad(u) * levelset.n) * (grad(v) * levelset.n) * dx(definedonelements=band)
    oldband = ifels

    ebext = ElementBasedExtension(gfu, oldband, band,energyform=energyform, activeelems=band, activefacets=None)
    ebext.Step()

    for i in range(9):
        oldband[:] = band
        AddNeighborhood(band, AdjacencyMatrix(mesh, "vertex"), layers=1, inplace=True)
        ebext.Step()
        assert Integrate((u_exact - gfu)**2 * dx(definedonelements=band), mesh)**(1/2) < 1e-10

        # region gets larger
        assert Integrate((BitArrayCF(band) - BitArrayCF(oldband)) * dx, mesh) > 0


