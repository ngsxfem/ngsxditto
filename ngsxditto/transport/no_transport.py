from .basetransport import *
from ngsolve import *


class NoTransport(BaseTransport):
    def __init__(self, mesh, order=2):
        super().__init__(mesh=mesh, wind=None, inflow_values=None, order=order)
        self.fes = H1(mesh, order=order)
        self.gfu = GridFunction(self.fes)

    def SetInitialValues(self, initial_values: CoefficientFunction=None, initial_time: float = 0.0):
        self.gfu.Set(initial_values)


    def SetTime(self, time):
        raise NotImplementedError("SetTime not implemented for NoTransport.")

    def UpdateStates(self):
        raise NotImplementedError("OneStep not implemented for NoTransport.")

    @property
    def field(self):
        return self.gfu

