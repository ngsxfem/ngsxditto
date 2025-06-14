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

    min_grad_old, max_grad_old = levelset.MinMaxGradientNorm()
    levelset.Redistance()
    min_grad_new, max_grad_new = levelset.MinMaxGradientNorm()
    assert min_grad_new > min_grad_old and max_grad_new < max_grad_old


def test_with_autoredistancing():
    transport = ImplicitSUPGTransport(mesh, wind, inflow_values=None, dt=dt, order=1)
    t.Set(0)
    transport.time = t
    transport.SetInitialValues(true_circle)

    redistancing = FastMarching()
    auto_redistancing = PeriodicRedistancing(100)
    levelset = LevelSetGeometry(transport, redistancing, auto_redistancing)

    levelset.transport.multistepper.RunFixedSteps(99)
    assert levelset.steps_since_last_redistancing == 99
    levelset.transport.OneStep()
    assert levelset.steps_since_last_redistancing == 0

    assert Integrate((levelset.transport.field - true_circle)**2, mesh)**(1/2) < 0.1

    levelset.RunUntilTime(transport.time.Get() + T_end)
    assert Integrate((levelset.transport.field - true_circle) ** 2, mesh) ** (1/2) < 0.1



