from ngsxditto.transport import *
from ngsolve import *
from netgen.geom2d import SplineGeometry
from xfem.lsetcurv import *


# Example: Rotating circle
domain = SplineGeometry()
domain.AddCircle((0,0), 1)
mesh = Mesh(domain.GenerateMesh(maxh=0.1))
wind = CF((-y, x))
t = Parameter(0)
true_circle = ((x - 1/2 * sin(t))**2 + (y + 1/2 * cos(t))**2)**(1/2) - 1/2

T_end = 1
dt = 0.05

def test_propagation():
    transport = KnownSolutionTransport(mesh, true_circle, dt=dt, order=3)
    transport.time = t
    t.Set(0)

    transport.Step()
    assert Integrate((transport.field - true_circle)**2, mesh)**(1/2) < 1e-3    # only interpolation error
    transport.multistepper.RunFixedSteps(10)
    assert Integrate((transport.field - true_circle)**2, mesh)**(1/2) < 1e-3
    transport.multistepper.RunUntilTime(2)
    assert Integrate((transport.field - true_circle)**2, mesh)**(1/2) < 1e-3
    transport.SetTime(5)
    assert Integrate((transport.field - true_circle)**2, mesh)**(1/2) < 1e-3



