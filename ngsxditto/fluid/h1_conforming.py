from ngsolve import *
from xfem import *
from .params import FluidParameters, WallParameters
from .discretization import FluidDiscretization
from ngsxditto.levelset import LevelSetGeometry, DummyLevelSet
from ngsxditto import direct_solver_spd, direct_solver_nonspd
from .meancurv import *
import ngsolve.webgui as ngw


class H1Conforming(FluidDiscretization):
    """
    This class handles all H1-conforming fluid discretizations.
    """
    def __init__(self, mesh, fluid_params: FluidParameters, order:int, lset:LevelSetGeometry,
                 wall_params: WallParameters, if_dirichlet:CoefficientFunction, add_convection:bool,
                 f: CoefficientFunction, g: CoefficientFunction,
                 surface_tension: CoefficientFunction, dt:float,
                 nitsche_stab:int, ghost_stab:int, extension_radius:float, derivative_jumps:bool, add_number_space:bool):
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
                         surface_tension=surface_tension, dt=dt, if_dirichlet=if_dirichlet, add_convection=add_convection,
                         derivative_jumps=derivative_jumps, add_number_space=add_number_space)
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
        self.EA = ElementAggregation(mesh)


    def SetLevelSet(self, lset):
        """
        Sets the levelset.
        """
        super().SetLevelSet(lset=lset)


    def SetInitialValues(self, initial_velocity, initial_pressure=CF(0), mean_pressure_fix=None):
        self.mesh.SetDeformation(self.lset.deformation)
        self.gfu.Set(initial_velocity)
        self.gfp.Set(initial_pressure)
        self.mesh.UnsetDeformation()

        self.ValidateStep()


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
        self.els_ring = self.els_outer & ~els_inner
        self.facets_ring = GetFacetsWithNeighborTypes(self.mesh, a=self.els_outer, b=self.els_ring)
        self.active_dofs = GetDofsOfElements(self.fes, self.els_outer)
        self.EA.Update(els_hasneg & ~self.lset.hasif, self.lset.hasif | (self.els_outer & ~ els_hasneg))

    def InitializeForms(self):
        self.AssembleAllForms()
        self.InvertTimeStepping()

    def AssembleAllForms(self):
        self.AssembleLf()

        if self.add_convection:
            self.AssembleConvection()

        self.AssembleStokes()

        self.AssembleTimeStepping()

    def AssembleLf(self):
        test = self.fes.TestFunction()
        v, q = test[0], test[1]
        s = test[2] if self.add_number_space else None

        h = specialcf.mesh_size
        n = self.lset.n

        dx_neg = self.lset.dx_neg
        dS = self.lset.dS
        self.lf = LinearForm(self.fes)
        self.lf += self.f * v * dx_neg
        self.lf += self.g * q * dx_neg

        if self.surface_tension is not None:
            self.lf += -self.fluid_params.surface_tension_coeff * self.surface_tension * v * dS

        if self.if_dirichlet is not None:
            self.lf += (-self.nu * grad(v) * n * self.if_dirichlet +
                        self.nu * self.nitsche_stab / h * self.if_dirichlet * v +
                        q * n * self.if_dirichlet) * dS

        for (region, fct) in self.neumann.items():
            self.lf += self.nu * fct * v * dx(definedon=self.mesh.Boundaries(region))

        self.lf.Assemble()

    def AssembleStokes(self):
        trial, test = self.fes.TnT()

        u, p = trial[0], trial[1]
        v, q = test[0], test[1]
        r = trial[2] if self.add_number_space else None
        s = test[2] if self.add_number_space else None

        h = specialcf.mesh_size
        n = self.lset.n

        dx_neg = self.lset.dx_neg
        dS = self.lset.dS

        basic_stokes = (self.nu * InnerProduct(grad(u), grad(v)) - 1/self.rho * p * div(v) - 1/self.rho * q * div(u)) * dx_neg

        if not self.derivative_jumps:
            #dw = dFacetPatch(definedonelements=self.facets_ring, deformation=self.lset.deformation)
            dw = dFacetPatch(definedonelements=self.EA.patch_interior_facets, deformation=self.lset.deformation)

            ghost_u = 1/h**2 * (u - u.Other()) * (v - v.Other()) * dw
            ghost_p = (p - p.Other()) * (q - q.Other()) * dw

        else:
            dw = dx(skeleton=True, definedonelements=self.facets_ring, deformation=self.lset.deformation)
            n_F = specialcf.normal(self.mesh.dim)
            ghost_p = h**3 * InnerProduct((grad(p) - grad(p.Other())) * n_F, (grad(q) - grad(q.Other())) * n_F) * dw

            ghost_u = h * InnerProduct((grad(u) - grad(u.Other())) * n_F, (grad(v) - grad(v.Other())) * n_F) * dw
            if self.order >=1:
                for i in range(self.mesh.dim):
                    ghost_u += h**3 * InnerProduct(
                        (u.Operator("hesse")[i] - u.Other().Operator("hesse")[i]) * n_F,
                        (v.Operator("hesse")[i] - v.Other().Operator("hesse")[i]) * n_F) * dw

        ghost_penalty = self.nu * self.ghost_stab * self.extension_radius * ghost_u - 1/self.nu * self.ghost_stab * ghost_p

        self.stokes_term = basic_stokes + ghost_penalty
        if self.if_dirichlet is not None:
            nitsche = (-grad(u) * n * v - grad(v) * n * u + self.nitsche_stab / h * u * v) * dS
            self.stokes_term += self.nu * nitsche
            self.stokes_term += (p * v * n + q * u * n) * dS

        if self.add_number_space:
            self.stokes_term += ((p * s + q * r) - (1e-8  * r * s)) * dx_neg
        else:
            self.stokes_term += 1e-10 * p * q * dx_neg

        self.stokes_op = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer,
                                                facet_restriction=self.facets_ring, check_unused=False)
        self.stokes_op += self.stokes_term
        self.stokes_op.Assemble(reallocate=True)


    def AssembleConvection(self):
        trial, test = self.fes.TnT()
        u, p = trial[0], trial[1]
        v, q = test[0], test[1]

        dx_neg = self.lset.dx_neg

        self.conv = ((grad(u) * self.intermediate.components[0]) * v * dx_neg +
                     (grad(self.intermediate.components[0]) * u) * v * dx_neg -
                     (grad(self.intermediate.components[0]) * self.intermediate.components[0]) * v * dx_neg)

        self.conv_op = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer, facet_restriction=self.facets_ring, check_unused=False)
        self.conv_op += self.conv
        self.conv_op.Assemble(reallocate=True)

    def AssembleTimeStepping(self):
        trial, test = self.fes.TnT()
        u, p = trial[0], trial[1]
        v, q = test[0], test[1]

        dx_neg = self.lset.dx_neg

        self.mass = u * v * dx_neg
        self.mass_op = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer, facet_restriction=self.facets_ring, check_unused=False)
        self.mass_op += self.mass
        self.mass_op.Assemble(reallocate=True)

        self.m_star = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer, facet_restriction=self.facets_ring, check_unused=False)
        self.m_star += self.mass + self.dt * self.stokes_term
        if self.add_convection:
            self.m_star += self.dt * self.conv
        self.m_star.Assemble(reallocate=True)

    def InvertTimeStepping(self):
        self.inv = self.m_star.mat.Inverse(freedofs=self.active_dofs & self.fes.FreeDofs(), inverse=direct_solver_nonspd)


    def SolveStokes(self):
        gfup = GridFunction(self.fes)
        gfu = gfup.components[0]
        default = CF((0,0)) if self.mesh.dim == 2 else CF((0,0,0))
        cf = self.mesh.BoundaryCF(self.dirichlet, default=default)
        gfu.Set(cf, definedon=self.mesh.Boundaries(self.dbnd))

        trial, test = self.fes.TnT()
        u, p = trial[0], trial[1]
        v, q = test[0], test[1]

        stationary_stokes_op = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer,
                                                facet_restriction=self.facets_ring, check_unused=False)
        stationary_stokes_op += self.stokes_term
        stationary_stokes_op += (1e-6 * u * v) * self.lset.dx_neg
        stationary_stokes_op.Assemble(reallocate=True)
        gfup.vec.data += (stationary_stokes_op.mat.Inverse(self.active_dofs & self.fes.FreeDofs(), inverse=direct_solver_nonspd) *
                         (self.lf.vec - stationary_stokes_op.mat * gfup.vec))
        return gfup


    def Step(self):
        if self.time is not None:
            self.time += self.dt
        self.AssembleLf()

        res = self.mass_op.mat * self.past.vec + self.dt * self.lf.vec - self.m_star.mat * self.gfup.vec
        self.gfup.vec.data += self.inv * res

        # gfup_copy = self.gfup.vec.CreateVector()
        # gfup_copy.data = self.gfup.vec
        #
        # self.gfup.vec[:] = 0
        # self.ApplyBoundaryConditions()
        #
        # uD = self.gfup.vec.CreateVector()
        # uD.data = self.gfup.vec
        #
        # self.gfup.vec.data += self.inv * (self.mass_op.mat*gfup_copy + self.dt * self.lf.vec - self.m_star.mat * uD)



    def SetTimeStepSize(self, dt):
        self.dt = dt
        self.InitializeForms()
