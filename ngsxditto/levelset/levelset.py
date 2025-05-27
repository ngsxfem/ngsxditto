from ngsolve import *
from ngsxditto.transport import *
from ngsxditto.redistancing import *


class LevelSetGeometry:
    def __init__(self, transport: BaseTransport, redistancing: BaseRedistancing):
        self.transport = transport
        self.redistancing = redistancing
        self.redistancing.SetOrder(transport.order)
        self.mesh = self.transport.mesh

    def Initialize(self, initial_lset: CoefficientFunction, initial_time: float=0.0):
        self.transport.SetInitialValues(initial_lset, initial_time)

    def OneStep(self):
        self.transport.OneStep()

    def Redistance(self):
        self.redistancing.Redistance(self.transport.field)

    def TestGradients(self, lower_bound, upper_bound, bandwidth=None):
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

        if min_grad < lower_bound or max_grad > upper_bound:
            return False
        return True
