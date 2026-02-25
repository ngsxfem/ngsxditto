from ngsxditto import *
from ngsolve import *
from netgen.geom2d import SplineGeometry
from xfem.lsetcurv import *
from xfem.utils import *
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


def test_propagation_with_trace():
    transport = ExplicitDGTransport(mesh, wind, inflow_values=None, dt=dt, compile=False)
    transport.time = t
    t.Set(0)
    transport.SetInitialValues(true_circle)
    while transport.time < T_end:
        transport.Step()
        transport.ValidateStep()

    assert Integrate((transport.gfu - true_circle)**2, mesh)**(1/2) < 1e-2

def test_propagation_without_trace():
    transport = ExplicitDGTransport(mesh, wind, inflow_values=None, dt=dt, usetrace=False, compile=False)
    transport.time = t
    t.Set(0)
    transport.SetInitialValues(true_circle)

    while transport.time < T_end:
        transport.Step()
        transport.ValidateStep()

    assert Integrate((transport.gfu - true_circle)**2, mesh)**(1/2) < 1e-2


def test_change_parameters():
    transport = ExplicitDGTransport(mesh, wind, inflow_values=None, dt=dt, order=2, compile=False)
    transport.time = t
    t.Set(0)
    transport.SetInitialValues(true_circle)

    for _ in range(10):
        transport.Step()
        transport.ValidateStep()

    assert Integrate((transport.field - true_circle) ** 2, mesh) ** (1 / 2) < 1e-2

    transport.SetTimeStepSize(0.01)

    for _ in range(10):
        transport.Step()
        transport.ValidateStep()

    assert pytest.approx(transport.time.Get()) == 0.3
    assert Integrate((transport.field - true_circle) ** 2, mesh) ** (1 / 2) < 1e-2

    transport.SetWind(-wind)

    for _ in range(30):
        transport.Step()
        transport.ValidateStep()

    t.Set(0)
    assert Integrate((transport.field - true_circle) ** 2, mesh) ** (1 / 2) < 1e-2



def test_narrow_band_propagation():
    t.Set(0)
    transport_elems = BitArray(mesh.ne)
    transport_elems[:] = True
    support_elems = BitArray(mesh.ne)

    adj = AdjacencyMatrix(mesh, "vertex")


    target_elems = BitArray(mesh.ne)
    target_elems[:] = True

    transport = ExplicitDGTransport(mesh, wind=wind, inflow_values=None, dt=dt, order=1, compile=False, usetrace=False,
                                    active_elements=transport_elems)

    levelset = LevelSetGeometry(transport)
    levelset.Initialize(true_circle)

    def UpdateElemMarker():
        support_elems[:] = AddNeighborhood(levelset.hasif, adj, layers=1, inplace=False)
        transport_elems[:] = AddNeighborhood(levelset.hasif, adj, layers=3, inplace=False)

    ebext = ElementBasedExtension(transport.past, support_elems, transport_elems)

    time_loop = TimeLoop(time=t, dt=dt, end_time=T_end)
    time_loop.Register(UpdateElemMarker, name="udpate transport elements")
    time_loop.Register(ebext, name="element based level set extension")
    time_loop.Register(levelset, name="levelset")
    time_loop()

    reduced_field = transport.field * BitArrayCF(support_elems)
    reduced_true_sol = true_circle * BitArrayCF(support_elems)

    l2_error = Integrate((reduced_field - reduced_true_sol)**2, mesh)**(1/2)
    assert l2_error < 1e-1


