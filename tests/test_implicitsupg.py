from ngsxditto.transport import *
from ngsolve import *
from netgen.geom2d import SplineGeometry
from xfem.lsetcurv import *
import pytest


# Example: Rotating circle
domain = SplineGeometry()
domain.AddCircle((0,0), 1)
mesh = Mesh(domain.GenerateMesh(maxh=0.2))
wind = CF((-y, x))
t = Parameter(0)
true_circle = ((x - 1/2 * sin(t))**2 + (y + 1/2 * cos(t))**2)**(1/2) - 1/2

T_end = 1
dt = 0.02


def test_propagation():
    transport = ImplicitSUPGTransport(mesh, wind, inflow_values=None, dt=dt, order=2)
    transport.time = t
    transport.SetInitialValues(true_circle)

    while transport.time < T_end:
        transport.OneStep()

    assert Integrate((transport.field - true_circle)**2, mesh)**(1/2) < 1e-2

def test_change_parameters():
    transport = ImplicitSUPGTransport(mesh, wind, inflow_values=None, dt=dt, order=2)
    transport.time = t
    t.Set(0)
    transport.SetInitialValues(true_circle)

    for _ in range(10):
        transport.OneStep()

    assert Integrate((transport.field - true_circle)**2, mesh)**(1/2) < 1e-2

    transport.SetTimeStepSize(0.01)

    for _ in range(10):
        transport.OneStep()

    assert pytest.approx(transport.time.Get()) == 0.3
    assert Integrate((transport.field - true_circle) ** 2, mesh) ** (1/2) < 1e-2

    transport.SetWind(-wind)

    for _ in range(30):
        transport.OneStep()

    t.Set(0)
    assert Integrate((transport.field - true_circle)**2, mesh)**(1/2) < 1e-2





