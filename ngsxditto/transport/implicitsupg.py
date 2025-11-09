from ngsolve import *
from .basetransport import BaseTransport
from ngsxditto import direct_solver_spd, direct_solver_nonspd


class ImplicitSUPGTransport(BaseTransport):
    """
    This class propagates a function along a given velocity field (wind) using the Crank-Nicolson scheme
    in time and the SUPG method as space discretization.
    """
    def __init__(self, mesh, wind=None, inflow_values=None, dt=0.01, order=2, source=None):
        """
        Initializes the transport object with the given parameters.

        Parameters:
        -----------
        mesh: Mesh
            The computational Mesh
        wind: CoefficientFunction
            The velocity field that transports the levelset
        inflow_values: CoefficientFunction
            The inflow boundary data
        time: Parameter
            reference to a Parameter for the time (to update depending coeffiecient function during propagate)
        dt: float
            The time step size for the transport.
        """
        super().__init__(mesh, wind, inflow_values, dt, source, order=order)

        self.fes = H1(mesh, order=order)
        self.u, self.v = self.fes.TnT()

        self.wind = wind
        self.gamma = None

        self.bfa = BilinearForm(self.fes, symmetric=False)
        self.rhs = BilinearForm(self.fes, symmetric=False)
        self.inv = None

        self.mass_term = None
        self.conv = None
        if wind is not None:
            self.SetWind(wind)

        self.gfu = GridFunction(self.fes)
        self.current = self.gfu
        self.past = GridFunction(self.gfu.space)
        self.intermediate = GridFunction(self.gfu.space)


    def SetInitialValues(self, initial_values: CoefficientFunction, initial_time: float = 0.0):
        if self.time is not None:
            self.time.Set(initial_time)
        self.gfu.Set(initial_values)

    def SetWind(self, wind: CoefficientFunction):
        self.wind = wind
        self.UpdateForms()

    def UpdateForms(self):
        u, v = self.u, self.v
        h = specialcf.mesh_size
        W = L2(self.mesh, order=0)
        gamma_gfu = GridFunction(W)
        gamma_gfu.Set(h / (2 * Norm(self.wind) + 10**(-5)))
        self.gamma = CoefficientFunction(gamma_gfu)

        self.mass_term = u * (v + self.gamma * self.wind * grad(v)) * dx
        self.conv = self.wind * grad(u) * (v + self.gamma * self.wind * grad(v)) * dx

        self.bfa = BilinearForm(self.fes, symmetric=False)
        self.bfa += self.mass_term
        self.bfa += self.dt/2 * self.conv

        self.rhs = BilinearForm(self.fes)
        self.rhs += self.mass_term
        self.rhs += -self.dt / 2 * self.conv

    def SetTimeStepSize(self, dt: float):
        self.dt = dt
        self.UpdateForms()


    def Step(self):
        self.bfa.Assemble()
        self.inv = self.bfa.mat.Inverse(self.fes.FreeDofs(), inverse=direct_solver_nonspd)
        self.rhs.Assemble()
        if self.time is not None:
            self.time.Set(self.time.Get() + self.dt)
        self.gfu.vec.data = self.inv @ self.rhs.mat * self.gfu.vec

    @property
    def field(self):
        return self.gfu