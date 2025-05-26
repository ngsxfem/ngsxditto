from ngsxditto.redistancing import *
from ngsolve import *
from netgen.geom2d import SplineGeometry
from xfem.lsetcurv import *


domain = SplineGeometry()
domain.AddCircle((0,0), 1)
mesh = Mesh(domain.GenerateMesh(maxh=0.1))
circle = x**2 + y**2 - 0.25
true_signed_distance = (x**2 + y**2)**(1/2) - 1/2


def test_h1_global():
    V = H1(mesh, order=1)
    phi = GridFunction(V)
    phi.Set(circle)

    redistancing = LinearFastMarching()
    redistancing.Redistance(phi)

    assert Integrate((phi - true_signed_distance)**2, mesh)**(1/2) < 1e-1  # L2-error

def test_bandwidth():
    V = H1(mesh, order=1)
    phi = GridFunction(V)
    phi.Set(circle)

    redistancing = LinearFastMarching(bandwidth=0.5)
    redistancing.Redistance(phi)
    norm_grad = Norm(grad(phi))
    assert abs(norm_grad(mesh(0, 0)) - 1) > 0.5
    assert abs(norm_grad(mesh(0.3, 0)) - 1) < 0.15
    assert abs(norm_grad(mesh(0.7, 0)) - 1) < 0.15
    assert abs(norm_grad(mesh(1, 0)) - 1) > 0.5

def test_l2_global():
    V = L2(mesh, order=1, dgjumps=True)
    phi = GridFunction(V)
    phi.Set(circle)

    redistancing = LinearFastMarching()
    redistancing.Redistance(phi)

    assert Integrate((phi - true_signed_distance)**2, mesh)**(1/2) < 1e-1  # L2-error



