from ngsxditto.transport import *
from ngsolve import *
from netgen.geom2d import SplineGeometry
from xfem.lsetcurv import *


# Example: Rotating circle
domain = SplineGeometry()
domain.AddCircle((0,0), 1)
mesh = Mesh(domain.GenerateMesh(maxh=0.1))
wind = CF((-y, x))

T_end = 1
dt = pi

def test_propagation():
    t = Parameter(0)
    true_circle = ((x - 1 / 2 * sin(t)) ** 2 + (y + 1 / 2 * cos(t)) ** 2) ** (1 / 2) - 1 / 2

    transport = KnownSolutionTransport(mesh, true_circle, dt=dt, order=2)
    transport.time=t
    t.Set(0)
    t += dt
    transport.Step()
    assert Integrate((transport.field - true_circle)**2, mesh)**(1/2) < 1e-3
    assert Integrate((transport.past - true_circle)**2, mesh)**(1/2) > 1

    transport.ValidateStep()
    assert Integrate((transport.past - true_circle)**2, mesh)**(1/2) < 1e-3



