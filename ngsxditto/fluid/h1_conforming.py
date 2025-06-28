from ngsolve import *

from .params import FluidParameters, WallParameters
from .discretization import FluidDiscretization


class H1Conforming(FluidDiscretization):
    def __init__(self, mesh, fluid_params: FluidParameters, order=4, levelset=None, wall_params: WallParameters=None, dt=None):
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, levelset=levelset, wall_params=wall_params, dt=dt)


    def SetInitialValues(self, initial_velocity, initial_pressure=CF(0)):
        self.gfu = GridFunction(self.fes)
        self.gfu.components[0].Set(initial_velocity)
        self.gfu.components[1].Set(initial_pressure)


    def InitializeForms(self, rhs: CoefficientFunction = None):
        (u, p), (v, q) = self.fes.TnT()
        X = self.fes

        self.mass = u * v * dx

        self.stokes = (self.nu * InnerProduct(grad(u), grad(v)) +
                       div(u) * q + div(v) * p - 1e-10 * p * q) * dx

        if rhs is None:
            rhs = CF((0, 0)) if self.mesh.dim == 2 else CF((0, 0, 0))

        g = CF(0)  # divergence constraint: I think we never want nonzero, due to mass conservation of our fluids?

        self.a = BilinearForm(X)
        self.a += self.stokes
        self.a.Assemble()

        self.lf = LinearForm(X)
        self.lf += rhs * v * dx + g * q * dx
        for (region, fct) in self.neumann.items():
            self.lf += self.nu * fct * v * dx(definedon=self.mesh.Boundaries(region))

        self.lf.Assemble()

        self.conv = BilinearForm(self.fes, nonassemble=True)
        self.conv += (Grad(u) * u) * v * dx

        self.m_star = BilinearForm(self.fes)
        self.m_star += self.mass + self.dt * self.stokes
        self.m_star.Assemble()

        self.inv = self.m_star.mat.Inverse(freedofs=self.fes.FreeDofs(), inverse="sparsecholesky")

