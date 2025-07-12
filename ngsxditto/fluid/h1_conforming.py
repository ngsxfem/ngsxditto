from ngsolve import *
from xfem import *
from .params import FluidParameters, WallParameters
from .discretization import FluidDiscretization
from ngsxditto.levelset import LevelSetGeometry, DummyLevelSet
import ngsolve.webgui as ngw


class H1Conforming(FluidDiscretization):
    def __init__(self, mesh, fluid_params: FluidParameters, order=4, lset:LevelSetGeometry=None, wall_params: WallParameters=None, dt=None, sigma=100, ghost_stab=20, delta=0.2):
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, lset=lset, wall_params=wall_params, dt=dt)
        self.active_dofs=None
        self.els_outer = None
        self.facets_ring = None
        self.ghost_stab = ghost_stab
        self.sigma = sigma    # nitsche stabilization
        self.delta = delta    # extension ring

        self.lset.AddCallback(self.UpdateActiveDofs)
        self.lset.AddCallback(self.InitializeForms)
        lsetp1_outer = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field - self.delta, lsetp1_outer)

        lsetp1_inner = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field + self.delta, lsetp1_inner)

        self.ci_main = CutInfo(self.mesh, self.lset.lsetp1)
        self.ci_inner = CutInfo(self.mesh, lsetp1_inner)
        self.ci_outer = CutInfo(self.mesh, lsetp1_outer)

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


    def InitializeForms(self, rhs: CoefficientFunction = None):
        (u, p), (v, q) = self.fes.TnT()
        X = self.fes
        h = specialcf.mesh_size

        if rhs is None:
            rhs = CF((0, 0)) if self.mesh.dim == 2 else CF((0, 0, 0))

        g = CF(0)  # divergence constraint: I think we never want nonzero, due to mass conservation of our fluids?
        self.lf = LinearForm(X)
        self.lf += (rhs * v + g * q) * self.lset.dx_neg

        for (region, fct) in self.neumann.items():
            self.lf += self.nu * fct * v * dx(definedon=self.mesh.Boundaries(region))

        self.lf.Assemble()

        #self.conv = BilinearForm(self.fes, nonassemble=True)
        #self.conv += (Grad(u) * u) * v * dx


        self.mass = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer, facet_restriction=self.facets_ring, check_unused=False)
        self.mass += u * v * self.lset.dx_neg
        self.mass.Assemble(reallocate=True)

        dw_u = dFacetPatch(definedonelements=self.facets_ring, deformation=self.lset.deformation)
        p_facets = GetFacetsWithNeighborTypes(self.mesh, a=self.lset.hasneg, b=self.lset.hasif)
        dw_p = dFacetPatch(definedonelements=p_facets, deformation=self.lset.deformation)

        nitsche = (-grad(u)* self.lset.n * v - grad(v)*self.lset.n * u + self.sigma/h * u * v) * self.lset.dS
        a_hn = self.nu * InnerProduct(grad(u), grad(v)) * self.lset.dx_neg + self.nu * nitsche
        b_hn = (-p * div(v) - q +div(u)) * self.lset.dx_neg + (p*v*self.lset.n + q*u*self.lset.n)*self.lset.dS

        i_hn = 1/(h**2) * (u - u.Other()) * (v - v.Other()) * dw_u
        j_hn = (p - p.Other()) * (q - q.Other()) * dw_p
        s_hn = self.ghost_stab * self.nu * i_hn + self.ghost_stab * 1/self.nu * i_hn - self.ghost_stab * 1/self.nu * j_hn

        self.stokes =  a_hn + b_hn + s_hn
        self.a = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer, facet_restriction=self.facets_ring, check_unused=False)
        self.a += self.stokes
        self.a.Assemble(reallocate=True)

        self.m_star = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer, facet_restriction=self.facets_ring, check_unused=False)
        self.m_star += u * v * self.lset.dx_neg + self.dt * self.stokes
        self.m_star.Assemble(reallocate=True)

        self.inv = self.m_star.mat.Inverse(freedofs=self.active_dofs & self.fes.FreeDofs(), inverse="pardiso")


    def SolveStokes(self):
        gfu = GridFunction(self.fes)
        cf = self.mesh.BoundaryCF(self.dirichlet, default=CF((0, 0)))
        gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))
        gfu.vec.data += self.a.mat.Inverse(self.active_dofs & self.fes.FreeDofs()) * (self.lf.vec - self.a.mat * gfu.vec)
        return gfu


    def OneStep(self):
        if self.time is not None:
            self.time += self.dt

        res = self.a.mat * self.gfu.vec
        self.gfu.vec.data -= self.dt * self.inv * res


