from ngsxditto.transport import *
from ngsolve import *
from netgen.geom2d import SplineGeometry
from xfem.lsetcurv import *


# Example: Rotating circle
domain = SplineGeometry()
domain.AddCircle((0,0), 1)
mesh = Mesh(domain.GenerateMesh(maxh=0.2))
wind = CF((-y, x))
t = Parameter(0)
true_circle = ((x - 1/2 * sin(t))**2 + (y + 1/2 * cos(t))**2)**(1/2) - 1/2

T_end = 1
timestepsize = 0.02


def test_propagation_with_trace():
    # L2 error after one rotation
    transport = ExplicitDGTransport(mesh, wind, inflow_values=None, timestepsize=timestepsize, compile=False)
    transport.time = t
    t.Set(0)
    transport.SetInitialValues(true_circle)

    transport.Propagate(0, T_end)

    assert Integrate((transport.gfu - true_circle)**2, mesh)**(1/2) < 1e-2

def test_propagation_without_trace():
    # L2 error after one rotation
    transport = ExplicitDGTransport(mesh, wind, inflow_values=None, timestepsize=timestepsize, usetrace=False, compile=False)
    transport.time = t
    t.Set(0)
    transport.SetInitialValues(true_circle)

    transport.Propagate(0, T_end)

    assert Integrate((transport.gfu - true_circle)**2, mesh)**(1/2) < 1e-2





