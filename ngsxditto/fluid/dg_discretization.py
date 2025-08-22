from ngsolve import *

from .discretization import FluidDiscretization
from .params import FluidParameters, WallParameters
from .hdiv_conforming import *
from ngsxditto.levelset import LevelSetGeometry


class BDMDG(HDivConforming):
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, lset:LevelSetGeometry = None,
                 wall_params: WallParameters = None, dt=None):
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, lset=lset, wall_params=wall_params, dt=dt)
        self.Sigmah = None
        self.Qh = None

    def InitializeSpaces(self):
        if self.dbnd is None:
            raise TypeError("self.dbnd is still None. Set Boundary conditions first.")
        self.Sigmah = HDiv(self.mesh, order=self.order, dirichlet=self.dbnd)
        self.Qh = L2(self.mesh, order=self.order - 1)
        Zh = FESpace([self.Sigmah, self.Qh], dgjumps=True)
        self.fes = Zh
        self.gfu = GridFunction(self.fes)

    def SetInitialValues(self, initial_velocity, initial_pressure=CF(0)):
        self.gfu = GridFunction(self.fes)
        self.gfu.components[0].Set(initial_velocity)
        self.gfu.components[1].Set(initial_pressure)

    def InitializeForms(self, rhs: CoefficientFunction = None, mean_curv = None):
        if mean_curv != None:
            raise Exception("BDM DG fluid got some mean curvature, this behaviour is not implemented!")

        n = specialcf.normal(self.mesh.dim)
        h = specialcf.mesh_size
        (self.u,  self.p), (self.v, self.q) = self.fes.TnT()
        (u, p), (v, q) = (self.u, self.p), (self.v, self.q)

        tang = lambda w: w - (w * n) * n
        grad_normal_average = lambda w: 1/2*(Grad(w)*n + Grad(w.Other())*n)
        tang_jump = lambda w: tang(w - w.Other())

        self.mass = u * v * dx

        self.stokes = (self.nu * InnerProduct(Grad(u), Grad(v)) - div(u) * q - div(v) * p - 1e-10 * p * q) * dx
        self.stokes += (-self.nu * grad_normal_average(u) * tang_jump(v) - self.nu * grad_normal_average(v) * tang_jump(u) +
                   self.lamb * self.nu/h * tang_jump(u)* tang_jump(v)) * dx(skeleton=True)
        self.stokes += (-self.nu*Grad(u)*n * tang(v) - self.nu*Grad(v)*n *tang(u) + self.lamb*self.nu/h *tang(u)*tang(v)) *ds(skeleton=True)

        self.a = BilinearForm(self.stokes).Assemble()

        g = CF(0)
        if rhs is None:
            rhs = CF((0, 0)) if self.mesh.dim == 2 else CF((0, 0, 0))
        self.lf = LinearForm(self.fes)
        self.lf += rhs * v * dx + g * q * dx

        for (region, fct) in self.neumann.items():
            self.lf += self.nu * fct * v * dx(definedon=self.mesh.Boundaries(region))


        self.m_star = BilinearForm(self.fes)
        self.m_star += self.rho * self.mass
        self.m_star += self.dt * self.stokes
        self.m_star.Assemble()
        self.inv = self.m_star.mat.Inverse(self.fes.FreeDofs(), "sparsecholesky")

        self.conv = BilinearForm(self.fes, nonassemble=True)
        self.conv += -self.rho*(Grad(v) *u) * u * dx
        self.conv += self.rho*u * n * IfPos(u*n, u, u.Other()) * v * dx(element_boundary=True)


