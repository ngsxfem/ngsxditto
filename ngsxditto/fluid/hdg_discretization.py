from ngsolve import *

from .discretization import FluidDiscretization
from .params import FluidParameters, WallParameters
from .hdiv_conforming import *
from ngsxditto.levelset import LevelSetGeometry


class BDMHDG(HDivConforming):
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, lset:LevelSetGeometry = None,
                 wall_params: WallParameters = None, dt=None):
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, lset=lset, wall_params=wall_params, dt=dt)
        self.Sigmah = None
        self.Fh = None
        self.Qh = None

    def InitializeSpaces(self):
        if self.dbnd is None:
            raise TypeError("self.dbnd is still None. Set Boundary conditions first.")
        self.Sigmah = HDiv(self.mesh, order=self.order, dirichlet=self.dbnd)
        self.Fh = TangentialFacetFESpace(self.mesh, order=self.order, dirichlet=self.dbnd)
        self.Qh = L2(self.mesh, order=self.order - 1)
        Zh = FESpace([self.Sigmah, self.Fh, self.Qh], dgjumps=False)
        self.fes = Zh
        self.gfu = GridFunction(self.fes)

    def SetInitialValues(self, initial_velocity, initial_pressure=CF(0)):
        self.gfu = GridFunction(self.fes)
        self.gfu.components[0].Set(initial_velocity)
        self.gfu.components[2].Set(initial_pressure)

    def InitializeForms(self, rhs: CoefficientFunction = None):
        n = specialcf.normal(self.mesh.dim)
        h = specialcf.mesh_size
        (self.uT, self.uF, self.p), (self.vT, self.vF, self.q) = self.fes.TnT()
        (uT, uF, p), (vT, vF, q) = (self.uT, self.uF, self.p), (self.vT, self.vF, self.q)

        tang = lambda uT: uT - (uT*n)*n
        facet_jump = lambda uT, uF: tang(uT - uF)

        self.mass = uT * vT * dx

        self.stokes = (self.nu * InnerProduct(Grad(uT), Grad(vT)) - div(uT) * q - div(vT) * p - 1e-10 * p * q) * dx
        self.stokes += (-self.nu * Grad(uT) * n * facet_jump(vT, vF) - self.nu * Grad(vT) * n * facet_jump(uT, uF) +
                   self.lamb * self.nu/h * facet_jump(uT, uF) * facet_jump(vT, vF)) * dx(element_boundary=True)

        self.a = BilinearForm(self.stokes).Assemble()
        g = CF(0)
        if rhs is None:
            rhs = CF((0, 0)) if self.mesh.dim == 2 else CF((0, 0, 0))
        self.lf = LinearForm(self.fes)
        self.lf += rhs * vT * dx + g * q * dx

        for (region, fct) in self.neumann.items():
            self.lf += self.nu * fct * vT * dx(definedon=self.mesh.Boundaries(region))

        self.m_star = BilinearForm(self.fes)
        self.m_star += self.rho * self.mass
        self.m_star += self.dt * self.stokes
        self.m_star.Assemble()
        self.inv = self.m_star.mat.Inverse(self.fes.FreeDofs(), "sparsecholesky")

        self.conv = BilinearForm(self.fes, nonassemble=True)
        self.conv += -self.rho*(Grad(vT) *uT) * uT * dx
        self.conv += self.rho*uT * n * IfPos(uT*n, uT, tang(uF)+(uT*n)*n) * vT * dx(element_boundary=True)
        self.conv += self.rho*IfPos(uT * n, uT * n * tang((uF - uT)) * (tang(vF) + (vT*n)*n), 0) * dx(element_boundary=True)

