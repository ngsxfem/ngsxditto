from ngsolve import *
from sympy.logic.boolalg import Boolean
from xfem import *
from .params import FluidParameters, WallParameters
from .discretization import FluidDiscretization
from ngsxditto.levelset import LevelSetGeometry, DummyLevelSet
from .meancurv import *
import ngsolve.webgui as ngw


class H1Conforming(FluidDiscretization):
    def __init__(self, mesh, fluid_params: FluidParameters, order=4, if_dirichlet:CoefficientFunction=None, lset:LevelSetGeometry=None, wall_params: WallParameters=None, dt=None, sigma=100, ghost_stab=20, delta=0.2):
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, lset=lset, wall_params=wall_params, dt=dt, if_dirichlet=if_dirichlet)
        self.active_dofs=None
        self.els_outer = None
        self.facets_ring = None
        self.ghost_stab = ghost_stab
        self.sigma = sigma    # nitsche stabilization
        self.delta = delta    # extension ring
        if lset is not None:
            self.lset.AddCallback(self.UpdateActiveDofs)
        lsetp1_outer = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field - self.delta, lsetp1_outer)

        lsetp1_inner = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field + self.delta, lsetp1_inner)

        self.ci_main = CutInfo(self.mesh, self.lset.lsetp1)
        self.ci_inner = CutInfo(self.mesh, lsetp1_inner)
        self.ci_outer = CutInfo(self.mesh, lsetp1_outer)


    def SetLevelSet(self, lset):
        super().SetLevelSet(lset=lset)
        if self.UpdateActiveDofs not in lset.callbacks:
            lset.callbacks.append(self.UpdateActiveDofs)


    def SetInitialValues(self, initial_velocity, initial_pressure=CF(0)):
        self.gfu = GridFunction(self.fes)
        self.gfu.components[0].Set(initial_velocity)
        self.gfu.components[1].Set(initial_pressure)


    def UpdateActiveDofs(self):
        lsetp1_outer = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field - self.delta, lsetp1_outer)

        lsetp1_inner = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field + self.delta, lsetp1_inner)

        self.ci_main.Update(self.lset.lsetp1)
        self.ci_inner.Update(lsetp1_inner)
        self.ci_outer.Update(lsetp1_outer)

        # Element and facet markers
        els_hasneg = self.ci_main.GetElementsOfType(HASNEG)
        self.els_outer = self.ci_outer.GetElementsOfType(HASNEG)
        els_inner = self.ci_inner.GetElementsOfType(NEG)
        els_ring = self.els_outer & ~els_inner
        self.facets_ring = GetFacetsWithNeighborTypes(self.mesh, a=self.els_outer, b=els_ring)
        self.active_dofs = GetDofsOfElements(self.fes, self.els_outer)

    def InitializeForms(self, rhs: CoefficientFunction = None, mean_curv=None):
        (u, p), (v, q) = self.fes.TnT()
        X = self.fes
        h = specialcf.mesh_size
        n = self.lset.n

        if rhs is None:
            rhs = CF((0, 0)) if self.mesh.dim == 2 else CF((0, 0, 0))

        dx_neg = self.lset.dx_neg
        dS = self.lset.dS

        self.lf = LinearForm(X)
        self.lf += self.rho * rhs * v * dx_neg
        if mean_curv is not None:
            self.lf += -self.fluid_params.surface_tension_coeff * mean_curv * v * dS
        if self.if_dirichlet is not None:
            self.lf += (-self.nu * Grad(v) * n * self.if_dirichlet + self.nu*self.sigma/h * self.if_dirichlet * v + q * n * self.if_dirichlet) * dS

        for (region, fct) in self.neumann.items():
            self.lf += self.nu * fct * v * dx(definedon=self.mesh.Boundaries(region))
        self.lf.Assemble()

        self.mass = u * v * dx_neg

        dw = dFacetPatch(definedonelements=self.facets_ring, deformation=self.lset.deformation)

        basic_stokes = (self.nu * InnerProduct(grad(u), grad(v)) - p * div(v) - q * div(u)) * dx_neg
        nitsche = (-grad(u)* n * v - grad(v)*n* u + self.sigma/h * u * v) * dS

        ghost_u = 1/h**2 * (u - u.Other()) * (v - v.Other()) * dw
        ghost_p = (p - p.Other()) * (q - q.Other()) * dw
        ghost_penalty = self.ghost_stab * self.nu * ghost_u + self.ghost_stab * 1/self.nu * ghost_u - self.ghost_stab * 1/self.nu * ghost_p

        self.stokes = basic_stokes + ghost_penalty
        if self.if_dirichlet is not None:
            self.stokes += nitsche
            self.stokes += (p*v*n + q*u*n)*dS

        if mean_curv is not None:
            self.stokes += 1e-5 * u * v * dx_neg

        self.a = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer, facet_restriction=self.facets_ring, check_unused=False)
        self.a += self.stokes
        self.a.Assemble(reallocate=True)

        self.m_star = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer, facet_restriction=self.facets_ring, check_unused=False)
        self.m_star += self.rho * self.mass + self.dt * self.stokes
        self.m_star.Assemble(reallocate=True)

        self.inv = self.m_star.mat.Inverse(freedofs=self.active_dofs & self.fes.FreeDofs())


    def SolveStokes(self):
        gfu = GridFunction(self.fes)
        cf = self.mesh.BoundaryCF(self.dirichlet, default=CF((0, 0)))
        gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))
        gfu.vec.data += self.a.mat.Inverse(self.active_dofs & self.fes.FreeDofs()) * (self.lf.vec - self.a.mat * gfu.vec)
        return gfu


    def OneStep(self):
        if self.time is not None:
            self.time += self.dt

        res = self.lf.vec - self.a.mat * self.gfu.vec
        self.gfu.vec.data += self.dt * self.inv * res


    def SetTimeStepSize(self, dt):
        self.dt = dt
        self.m_star = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer, facet_restriction=self.facets_ring, check_unused=False)
        self.m_star += self.mass + self.dt * self.stokes
        self.m_star.Assemble(reallocate=True)
        self.inv = self.m_star.mat.Inverse(freedofs=self.active_dofs & self.fes.FreeDofs())

