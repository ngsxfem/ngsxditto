from ngsolve import *

from .multistepper import MultiStepper
from ngsxditto.transport import *
from ngsxditto.redistancing import *


class LevelSetGeometry:
    def __init__(self, transport: BaseTransport, redistancing: BaseRedistancing, multistepper: MultiStepper, autoredistancing: AutoRedistancing=None):
        self.transport = transport
        self.redistancing = redistancing
        self.redistancing.SetOrder(transport.order)
        self.mesh = self.transport.mesh
        self.multistepper = multistepper
        self.multistepper.SetLevelSet(self)
        self.autoredistancing = autoredistancing
        self.steps_since_last_redistancing = 0

    def Initialize(self, initial_lset: CoefficientFunction, initial_time: float=0.0):
        self.transport.SetInitialValues(initial_lset, initial_time)

    def OneStep(self):
        self.transport.OneStep()
        self.steps_since_last_redistancing += 1

    def ShouldRedistance(self):
        if self.autoredistancing is not None:
            return self.autoredistancing.ShouldRedistance(self)


    def Redistance(self):
        print("The next function is redistanced")
        self.redistancing.Redistance(self.transport.field)
        self.steps_since_last_redistancing = 0


    def MinMaxGradientNorm(self, bandwidth=None):
        phi = self.transport.field
        norm_grad = Norm(grad(phi))
        V = phi.space
        gfu = GridFunction(V)
        gfu.Set(norm_grad)

        max_grad = -1e100
        min_grad = 1e100

        for v in self.mesh.vertices:
            point = self.mesh[v].point
            val = gfu(self.mesh(*point))
            max_grad = max(max_grad, val)
            min_grad = min(min_grad, val)

        return min_grad, max_grad
