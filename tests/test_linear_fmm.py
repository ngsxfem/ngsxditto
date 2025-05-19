from sympy.physics.quantum import L2

from ngsxditto.redistancing import *
from ngsolve import *
from netgen.geom2d import SplineGeometry
from xfem.lsetcurv import *


domain = SplineGeometry()
domain.AddCircle((0,0), 1)
mesh = Mesh(domain.GenerateMesh(maxh=0.2))
circle = x**2 + y**2 - 0.25
true_signed_distance = (x**2 + y**2)**(1/2) - 1/2


def test_h1_global():
    V = H1(mesh, order=1)
    phi = GridFunction(V)
    phi.Set(circle)

    redistancing = LinearFastMarching()
    redistancing.Redistance(phi)

    assert Integrate((phi - true_signed_distance)**2, mesh)**(1/2) < 1e-1  # L1-error


def test_l2_global():
    V = L2(mesh, order=1, dgjumps=True)
    phi = GridFunction(V)
    phi.Set(circle)

    redistancing = LinearFastMarching()
    redistancing.Redistance(phi)

    assert Integrate((phi - true_signed_distance)**2, mesh)**(1/2) < 1e-1  # L1-error

