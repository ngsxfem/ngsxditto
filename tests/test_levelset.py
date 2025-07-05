from ngsxditto.levelset import *
from ngsolve import *
from netgen.geom2d import SplineGeometry


domain = SplineGeometry()
domain.AddCircle((0,0), 1)
mesh = Mesh(domain.GenerateMesh(maxh=0.2))
wind = CF((-y, x))
t = Parameter(0)
true_circle = ((x - 1/2 * sin(t))**2 + (y + 1/2 * cos(t))**2)**(1/2) - 1/2

T_end = 1
dt = 0.02

def test_without_autoredistancing():
    transport = ImplicitSUPGTransport(mesh, wind, inflow_values=None, dt=dt, order=1)
    t.Set(0)
    transport.time = t
    transport.SetInitialValues(true_circle)

    redistancing = FastMarching()
    levelset = LevelSetGeometry(transport, redistancing)
    while transport.time < T_end:
        levelset.OneStep()
    assert Integrate((levelset.transport.field - true_circle)**2, mesh)**(1/2) < 0.1

    levelset.RunFixedSteps(50)
    assert Integrate((levelset.transport.field - true_circle)**2, mesh)**(1/2) < 0.1

    levelset.RunUntilTime(transport.time.Get() + T_end)
    assert Integrate((levelset.transport.field - true_circle)**2, mesh) ** (1/2) < 0.1


def test_with_autoredistancing():
    transport = ImplicitSUPGTransport(mesh, wind, inflow_values=None, dt=dt, order=1)
    t.Set(0)
    transport.time = t
    transport.SetInitialValues(true_circle)

    redistancing = FastMarching()
    auto_redistancing = PeriodicRedistancing(100)
    levelset = LevelSetGeometry(transport, redistancing, auto_redistancing)

    levelset.multistepper.RunFixedSteps(99)
    assert levelset.steps_since_last_redistancing == 99
    levelset.OneStep()
    assert levelset.steps_since_last_redistancing == 0

    assert Integrate((levelset.transport.field - true_circle)**2, mesh)**(1/2) < 0.1

    levelset.RunUntilTime(transport.time.Get() + T_end)
    assert Integrate((levelset.transport.field - true_circle) ** 2, mesh) ** (1/2) < 0.1

def test_dummy_levelset():
    dummy_lset = DummyLevelSet(mesh)
    dummy_lset.OneStep()
    assert Integrate((dummy_lset.field + 1)**2, mesh)**(1/2) < 1e-10
    dummy_lset.multistepper.RunFixedSteps(100)
    assert Integrate((dummy_lset.field + 1)**2, mesh)**(1/2) < 1e-10
