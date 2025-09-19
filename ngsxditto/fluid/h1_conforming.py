from ngsolve import *
from xfem import *
from .params import FluidParameters, WallParameters
from .discretization import FluidDiscretization
from ngsxditto.levelset import LevelSetGeometry, DummyLevelSet
from .meancurv import *
import ngsolve.webgui as ngw


class H1Conforming(FluidDiscretization):
    """
    This class handles all H1-conforming fluid discretizations.
    """
    def __init__(self, mesh, fluid_params: FluidParameters, order=4, lset:LevelSetGeometry=None,
                 wall_params: WallParameters = None, if_dirichlet:CoefficientFunction=None,
                 f: CoefficientFunction = CF((0, 0)), g: CoefficientFunction = CF(0),
                 surface_tension: CoefficientFunction = CF((0, 0)), dt=None,
                 nitsche_stab:int=100, ghost_stab:int=20, extension_radius:float=0.2):
        """
        Initializes the fluid discretization with the given parameters and levelset.
        Parameters:
        ----------
        mesh: Mesh
            The computational mesh
        fluid_params: FluidParameters
            parameter of fluid, like viscosity, density and surface tension coefficient.
        order: int
            the polynomial order
        lset: LevelsetGeometry
            The levelset that characterizes the unfitted domain.
        wall_params: WallParameters
            wall parameters for contact problems
        if_dirichlet: CoefficientFunction
            Dirichlet boundary condition of the unfitted domain.
        f: CoefficientFunction
            The force term
        g: CoefficientFunction
            The divergence constraint
        surface_tension: CoefficientFunction
            The surface tension force.
        dt: float
            Time-step size
        nitsche_stab: int
            The stabilization parameter for the nitsche term
        ghost_stab: int
            The ghost stability parameter
        extension_radius: float
            Radius of the zero levelset on which the domain is extended.
        """
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, lset=lset, wall_params=wall_params, f=f, g=g,
                         surface_tension=surface_tension, dt=dt, if_dirichlet=if_dirichlet)
        self.active_dofs=None
        self.els_outer = None
        self.facets_ring = None
        self.ghost_stab = ghost_stab
        self.nitsche_stab = nitsche_stab    # nitsche stabilization
        self.extension_radius = extension_radius    # extension ring

        lsetp1_outer = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field - self.extension_radius, lsetp1_outer)

        lsetp1_inner = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field + self.extension_radius, lsetp1_inner)

        self.ci_main = CutInfo(self.mesh, self.lset.lsetp1)
        self.ci_inner = CutInfo(self.mesh, lsetp1_inner)
        self.ci_outer = CutInfo(self.mesh, lsetp1_outer)


    def SetLevelSet(self, lset):
        """
        Sets the levelset.
        """
        super().SetLevelSet(lset=lset)


    def SetInitialValues(self, initial_velocity, initial_pressure=CF(0)):
        self.gfu = GridFunction(self.fes)
        self.gfu.components[0].Set(initial_velocity)
        self.gfu.components[1].Set(initial_pressure)
        self.StoreState()


    def UpdateActiveDofs(self):
        """
        Updates the dofs that are active, i.e. all dofs that are in the extended unfitted domain.
        """
        lsetp1_outer = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field - self.extension_radius, lsetp1_outer)

        lsetp1_inner = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field + self.extension_radius, lsetp1_inner)

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

    def InitializeForms(self):
        (u, p), (v, q) = self.fes.TnT()
        X = self.fes
        h = specialcf.mesh_size
        n = self.lset.n

        dx_neg = self.lset.dx_neg
        dS = self.lset.dS

        self.lf = LinearForm(X)
        self.lf += self.rho * self.f * v * dx_neg
        self.lf += self.g * q * dx_neg
        if self.surface_tension is not None:
            self.lf += -self.fluid_params.surface_tension_coeff * self.surface_tension * v * dS
        if self.if_dirichlet is not None:
            self.lf += (-self.nu * Grad(v) * n * self.if_dirichlet + self.nu * self.nitsche_stab / h * self.if_dirichlet * v + q * n * self.if_dirichlet) * dS

        for (region, fct) in self.neumann.items():
            self.lf += self.nu * fct * v * dx(definedon=self.mesh.Boundaries(region))
        self.lf.Assemble()

        self.mass = u * v * dx_neg

        dw = dFacetPatch(definedonelements=self.facets_ring, deformation=self.lset.deformation)

        basic_stokes = (self.nu * InnerProduct(grad(u), grad(v)) - p * div(v) - q * div(u)) * dx_neg
        nitsche = (-grad(u) * n * v - grad(v) * n * u + self.nitsche_stab / h * u * v) * dS

        ghost_u = 1/h**2 * (u - u.Other()) * (v - v.Other()) * dw
        ghost_p = (p - p.Other()) * (q - q.Other()) * dw
        ghost_penalty = self.ghost_stab * self.nu * ghost_u - self.ghost_stab * 1/self.nu * ghost_p + self.ghost_stab * 1/self.nu * ghost_u

        self.stokes = basic_stokes + ghost_penalty
        if self.if_dirichlet is not None:
            self.stokes += nitsche
            self.stokes += (p*v*n + q*u*n)*dS

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


    def OneStep(self, finalize=True):
        if self.time is not None:
            self.time += self.dt

        self.gfu.vec.data = self.past
        res = self.lf.vec - self.a.mat * self.gfu.vec
        self.gfu.vec.data += self.dt * self.inv * res

        #if self.intermediate_valid:
            #self.intermediate_difference = self.ComputeDifference2Intermediate()
        #self.StoreIntermediate()
        if finalize:
            self.StoreState()

    def OneStepNoFinalize(self):
        self.OneStep(finalize=False)

    def SetTimeStepSize(self, dt):
        self.dt = dt
        self.m_star = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer, facet_restriction=self.facets_ring, check_unused=False)
        self.m_star += self.mass + self.dt * self.stokes
        self.m_star.Assemble(reallocate=True)
        self.inv = self.m_star.mat.Inverse(freedofs=self.active_dofs & self.fes.FreeDofs())


    def StoreState(self):
        self.past[:] = self.gfu.vec
        #self.intermediate_valid = False