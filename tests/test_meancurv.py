from ngsolve import *
from ngsxditto import *


domain = MoveTo(-1, -1).Rectangle(2, 2).Face()
domain.edges.Max(X).name = "right"
domain.edges.Min(X).name = "left"
domain.edges.Min(Y).name = "bottom"
domain.edges.Max(Y).name = "top"
mesh = Mesh(OCCGeometry(domain, dim=2).GenerateMesh(maxh=0.1))

r = 0.5
levelset_function = (x**2 + y**2)**(1/2) - r

levelset = LevelSetGeometry.from_cf(levelset_function, mesh, order=2)

def test_meancurv():
    mean_curvature = MeanCurvatureSolver(mesh, order=1, lset=levelset)
    mean_curvature.Step()

    norm = Norm(mean_curvature.H)
    true_mean_curv = 1/r

    assert Integrate((norm- true_mean_curv)**2 * levelset.dS, mesh)**(1/2) < 1e-2
    curv_x = mean_curvature.H[0]
    curv_y = mean_curvature.H[1]
    dS = levelset.dS
    assert abs(Integrate(curv_x * dS, mesh)) < 1e-2
    assert abs(Integrate(curv_y * dS, mesh)) < 1e-2



