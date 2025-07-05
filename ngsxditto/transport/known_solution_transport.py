from .basetransport import *
from ngsolve import *


class KnownSolutionTransport(BaseTransport):
    def __init__(self, mesh, true_solution:CoefficientFunction, time:Parameter=None, dt=None, order=2):
        super().__init__(mesh=mesh, wind=None, inflow_values=None, dt=dt, time=time, order=order)
        self.true_solution = true_solution
        self.fes = H1(mesh, order=order)
        self.gfu = GridFunction(self.fes)
        self.SetInitialValues(true_solution)


    def SetTimeStepSize(self, dt: float):
        self.dt = dt


    def SetTime(self, time):
        self.time.Set(time)
        self.gfu.Set(self.true_solution)

    def OneStep(self):
        if self.time is not None:
            self.time += self.dt
        self.gfu.Set(self.true_solution)

    @property
    def field(self):
        return self.gfu

