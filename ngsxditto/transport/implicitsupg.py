from ngsolve import *
from .basetransport import BaseTransport


class ImplicitSUPGTransport(BaseTransport):
    """
    This class propagates a function along a given velocity field (wind) using the Crank-Nicolson scheme
    in time and the SUPG method as space discretization.
    """
    def __init__(self, mesh, wind, inflow_values, dt, order=2, source=None):
        super().__init__(mesh, wind, inflow_values, dt, source, order=order)

        self.fes = H1(mesh, order=order)
        self.u, self.v = self.fes.TnT()
        u, v = self.u, self.v

        wind = self.wind

        h = specialcf.mesh_size
        W = L2(mesh, order=0)
        gamma_gfu = GridFunction(W)
        gamma_gfu.Set(h / (2 * Norm(wind) + 10**(-5)))
        self.gamma = CoefficientFunction(gamma_gfu)

        self.bfa = BilinearForm(self.fes, symmetric=False)
        self.rhs = BilinearForm(self.fes, symmetric=False)
        self.inv = None

        self.mass_term = u * (v + self.gamma * wind * grad(v)) * dx
        self.conv = wind * grad(u) * (v + self.gamma * wind * grad(v)) * dx

        self.SetWind(wind)

        self.gfu = GridFunction(self.fes)


    def SetInitialValues(self, initial_values: CoefficientFunction, initial_time: float = 0.0):
        if self.time is not None:
            self.time.Set(initial_time)
        self.gfu.Set(initial_values)

    def SetWind(self, wind: CoefficientFunction):
        self.wind = wind
        u, v = self.u, self.v
        self.mass_term = u * (v + self.gamma * wind * grad(v)) * dx
        self.conv = wind * grad(u) * (v + self.gamma * wind * grad(v)) * dx

        self.bfa = BilinearForm(self.fes, symmetric=False)
        self.bfa += self.mass_term
        self.bfa += self.dt/2 * self.conv
        self.bfa.Assemble()

        self.inv = self.bfa.mat.Inverse(self.fes.FreeDofs())
        self.rhs = BilinearForm(self.fes, symmetric=False)
        self.rhs += self.mass_term
        self.rhs += -self.dt / 2 * self.conv
        self.rhs.Assemble()

    def SetTimeStepSize(self, dt: float):
        self.dt = dt
        self.bfa = BilinearForm(self.fes, symmetric=False)
        self.bfa += self.mass_term
        self.bfa += self.dt/2 * self.conv
        self.bfa.Assemble()

        self.inv = self.bfa.mat.Inverse(self.fes.FreeDofs())
        self.rhs = BilinearForm(self.fes, symmetric=False)
        self.rhs += self.mass_term
        self.rhs += -self.dt / 2 * self.conv
        self.rhs.Assemble()


    def OneStep(self):
        if self.time is not None:
            self.time.Set(self.time.Get() + self.dt)
        self.gfu.vec.data = self.inv @ self.rhs.mat * self.gfu.vec

    @property
    def field(self):
        return self.gfu