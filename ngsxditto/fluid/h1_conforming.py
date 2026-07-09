import logging
from ngsolve import *
from xfem import *
from .params import FluidParameters, WallParameters
from .discretization import FluidDiscretization
from ngsxditto.levelset import LevelSetGeometry, DummyLevelSet
from ngsxditto import direct_solver_spd, direct_solver_nonspd
from .meancurv import *
import ngsolve.webgui as ngw

logger = logging.getLogger(__name__)


class H1Conforming(FluidDiscretization):
    """
    This class handles all H1-conforming fluid discretizations.
    """
    def __init__(self, mesh, fluid_params: FluidParameters, order:int, lset:LevelSetGeometry,
                 wall_params: WallParameters, add_convection:bool, f: CoefficientFunction, g: CoefficientFunction,
                 surface_tension: CoefficientFunction, dt:float, nitsche_stab:int, ghost_stab:int,
                 extension_radius:float, derivative_jumps:bool, add_number_space:bool, time_order:int, use_supg:bool):
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
        add_convection: bool
            Whether to add the convection term to the discretization. If False, solve the Stokes problem.
            If True, solve the Navier-Stokes problem.
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
        derivative_jumps: bool
            Whether to use the derivative jump ghost penalty (not recommended). Only implemented for order <= 2.
        add_number_space: bool
            Whether to add a number space to the finite element space for pressure stabilization.
        time_order: int
            The order of the time discretization. Only implemented up to order 2.
        use_supg: bool
            Whether to use SUPG stabilization for the convection term. (Not yet consistent.)
        """
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, lset=lset, wall_params=wall_params, f=f, g=g,
                         surface_tension=surface_tension, dt=dt, add_convection=add_convection,
                         derivative_jumps=derivative_jumps, add_number_space=add_number_space, time_order=time_order,
                         use_supg=use_supg)
        self.active_dofs=None
        self.els_outer = None
        self.facets_ring = None
        self.ghost_stab = ghost_stab
        self.nitsche_stab = nitsche_stab
        self.extension_radius = extension_radius

        lsetp1_outer = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field - self.extension_radius, lsetp1_outer)

        lsetp1_inner = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field + self.extension_radius, lsetp1_inner)

        self.ci_main = CutInfo(self.mesh, self.lset.lsetp1)
        self.ci_inner = CutInfo(self.mesh, lsetp1_inner)
        self.ci_outer = CutInfo(self.mesh, lsetp1_outer)
        self.EA = ElementAggregation(mesh)


    def _restrict_to_rooted_components(self, band, seeds):
        """
        Returns the elements of `band` whose facet-connected component contains
        an element of `seeds`. Far away from the interface the level set field
        is not controlled (it is advected by an extension velocity and not
        redistanced), so its perturbations can cross the extension-radius
        offset and create small band islands detached from the fluid domain.
        Such islands cannot be element-aggregated (no root element) and are
        irrelevant for the unfitted discretization, so they are dropped here.
        """
        import os
        if os.environ.get("NGSXDITTO_DISABLE_MARKER_FILTERS"):
            return BitArray(band)
        reached = seeds & band
        while True:
            front = GetFacetsWithNeighborTypes(self.mesh, a=reached, b=band)
            grown = GetElementsWithNeighborFacets(self.mesh, front)
            new = grown & band & ~reached
            if new.NumSet() == 0:
                break
            reached |= new
        n_dropped = (band & ~reached).NumSet()
        if n_dropped > 0:
            logger.debug("dropped %d detached extension-band element(s)", n_dropped)
        return reached

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
        if hasattr(self.lset, "hasneg"):
            # consistent with the (island-filtered) level set markings that
            # the integrators are defined on
            els_hasneg = els_hasneg & self.lset.hasneg
        roots = els_hasneg & ~self.lset.hasif
        filtered_outer = self._restrict_to_rooted_components(
            self.ci_outer.GetElementsOfType(HASNEG), roots)
        if self.els_outer is None:
            self.els_outer = filtered_outer
        else:
            # update in place: lsetadap.ProjectOnUpdate holds a reference to
            # this BitArray as its update domain
            self.els_outer &= filtered_outer
            self.els_outer |= filtered_outer
        els_inner = self.ci_inner.GetElementsOfType(NEG)
        els_ring = self.els_outer & ~els_inner
        self.facets_ring = GetFacetsWithNeighborTypes(self.mesh, a=self.els_outer, b=els_ring)
        self.active_dofs = GetDofsOfElements(self.fes, self.els_outer)
        try:
            self.EA.Update(roots, (self.lset.hasif | (self.els_outer & ~ els_hasneg)) & self.els_outer)
            self.ghost_facets = self.EA.patch_interior_facets
        except Exception as e:
            # aggregation of the current cut configuration can fail (level set
            # perturbations can create patch layouts without reachable roots);
            # fall back to stabilizing all ring facets for this step, which is
            # the classic (more conservative) ghost-penalty facet set
            logger.warning("ElementAggregation failed (%s); using ring-facet "
                           "ghost penalty for this step", e)
            self.ghost_facets = self.facets_ring

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
        n_bnd = specialcf.normal(self.mesh.dim)
        n_lset = self.lset.n
        if self.mesh.dim == 2:
            t = specialcf.tangential(2)
            n_line = IfPos(InnerProduct(t, n_lset), t, -t)
        else:
            # 3D: project interface normal onto the substrate surface to get the
            # spreading direction (co-normal to the contact line on the boundary)
            n_lset_surf = n_lset - InnerProduct(n_lset, n_bnd) * n_bnd
            n_line = n_lset_surf / (Norm(n_lset_surf) + 1e-12)

        dx_neg = self.lset.dx_neg
        dS = self.lset.dS
        self.lf = LinearForm(self.fes)
        self.lf += self.f * v * dx_neg
        self.lf += self.g * q * dx_neg
        if self.add_convection:
            u_approx = self.intermediate.components[0]
            self.lf += (grad(u_approx) * u_approx) * v * self.lset.dx_neg
        tau = self.fluid_params.surface_tension_coeff
        if self.surface_tension is not None:
            self.lf += - 1/self.rho * tau * self.surface_tension * v * dS

        for (region, values) in self.boundary_registry.nitsche_normal_velocity_dict.items():
            if region != "interface":
                self.lf += (-self.nu * (2*Sym(grad(v).Trace()) * n_bnd) * n_bnd * values + q * values
                            + 2* self.nu * self.nitsche_stab/h * (v * n_bnd) * values) * ds(definedon=self.mesh.Boundaries(region))
            else:
                self.lf += (-self.nu * (2*Sym(grad(v)) * n_lset) * n_lset * values + q * values
                            + 2 * self.nu * self.nitsche_stab / h * (v * n_lset) * values) * dS

        for (region, values) in self.boundary_registry.nitsche_velocity_dict.items():
            if region != "interface":
                self.lf += (-self.nu * 2*Sym(grad(v).Trace()) * n_bnd * values +
                            2 * self.nu * self.nitsche_stab / h * values * v +
                            q * n_bnd * values) * ds(definedon=self.mesh.Boundaries(region))
            else:
                self.lf += (-self.nu * 2*Sym(grad(v)) * n_lset * values +
                            2 * self.nu * self.nitsche_stab / h * values * v +
                            q * n_lset * values) * dS

        for (region, values) in self.boundary_registry.strong_neumann_dict.items():
            self.lf += values * v * dx(definedon=self.mesh.Boundaries(region))


        d_contact_plane = dCut(self.lset.lsetp1, domain_type=NEG,
                               deformation=self.lset.deformation, vb=BND)
        d_contact_line = dCut(self.lset.lsetp1, domain_type=IF,
                               deformation=self.lset.deformation, vb=BND)
        theta_e = self.wall_params.contact_angle

        self.lf += 1/self.rho * cos(theta_e) * tau * v * n_line * d_contact_line
        P_Gamma = Id(self.mesh.dim) - OuterProduct(n_lset, n_lset)
        P_S = Id(self.mesh.dim) - OuterProduct(n_bnd, n_bnd)
        eta_L = (P_Gamma * n_bnd)/Norm(P_Gamma * n_bnd)
        self.lf += -1/self.rho * P_S * tau * P_Gamma * eta_L * v * d_contact_line

        self.lf.Assemble()

    @timed_method
    def AssembleStokes(self):
        trial, test = self.fes.TnT()

        u, p = trial[0], trial[1]
        v, q = test[0], test[1]
        r = trial[2] if self.add_number_space else None
        s = test[2] if self.add_number_space else None

        h = specialcf.mesh_size
        n_bnd = specialcf.normal(self.mesh.dim)
        n_lset = self.lset.n

        dx_neg = self.lset.dx_neg
        dS = self.lset.dS

        basic_stokes = (2*self.nu * InnerProduct(Sym(grad(u)), Sym(grad(v))) - p * div(v) - q * div(u)) * dx_neg

        if not self.derivative_jumps:
            #dw = dFacetPatch(definedonelements=self.facets_ring, deformation=self.lset.deformation)
            dw = dFacetPatch(definedonelements=self.ghost_facets, deformation=self.lset.deformation)

            ghost_u = 1/h**2 * (u - u.Other()) * (v - v.Other()) * dw
            ghost_p = (p - p.Other()) * (q - q.Other()) * dw

        else:
            dw = dx(skeleton=True, definedonelements=self.facets_ring, deformation=self.lset.deformation)
            n_F = specialcf.normal(self.mesh.dim)
            ghost_p = h**3 * InnerProduct((grad(p) - grad(p.Other())) * n_F, (grad(q) - grad(q.Other())) * n_F) * dw

            ghost_u = h * InnerProduct((grad(u) - grad(u.Other())) * n_F, (grad(v) - grad(v.Other())) * n_F) * dw
            for i in range(self.mesh.dim):
                ghost_u += h**3 * InnerProduct(
                    (u.Operator("hesse")[i] - u.Other().Operator("hesse")[i]) * n_F,
                    (v.Operator("hesse")[i] - v.Other().Operator("hesse")[i]) * n_F) * dw

        ghost_penalty = self.nu * self.ghost_stab * self.extension_radius * ghost_u - 1/self.nu * self.ghost_stab * ghost_p

        self.stokes_term = basic_stokes + ghost_penalty

        for (region, values) in self.boundary_registry.nitsche_normal_velocity_dict.items():
            if region != "interface":
                un = u * n_bnd
                vn = v * n_bnd

                nitsche = (-(2*Sym(grad(u).Trace()) * n_bnd) * n_bnd * vn - (2*Sym(grad(v).Trace()) * n_bnd) * n_bnd * un
                          + 2 * self.nitsche_stab / h * un * vn) * ds(definedon=self.mesh.Boundaries(region))
                self.stokes_term += self.nu * nitsche
                self.stokes_term += (q * u * n_bnd + p * v * n_bnd) * ds(definedon=self.mesh.Boundaries(region))
            else:
                un = u * n_lset
                vn = v * n_lset

                nitsche = (-(2*Sym(grad(u)) * n_lset) * n_lset * vn - (2*Sym(grad(v)) * n_lset) * n_lset * un
                          + 2 * self.nitsche_stab / h * un * vn) * dS
                self.stokes_term += self.nu * nitsche
                self.stokes_term += (q * u * n_lset + p * v * n_lset) * dS


        for (region, values) in self.boundary_registry.nitsche_velocity_dict.items():
            if region != "interface":
                nitsche = (-2*Sym(grad(u).Trace()) * n_bnd * v - 2*Sym(grad(v).Trace()) * n_bnd * u + 2 * self.nitsche_stab / h * u * v)  * ds(definedon=self.mesh.Boundaries(region))
                self.stokes_term += self.nu * nitsche
                self.stokes_term += (p * v * n_bnd + q * u * n_bnd) * ds(definedon=self.mesh.Boundaries(region))

            else:
                nitsche = (-2*Sym(grad(u)) * n_lset * v - 2*Sym(grad(v)) * n_lset * u + 2 * self.nitsche_stab / h * u * v) * dS
                self.stokes_term += self.nu * nitsche
                self.stokes_term += (p * v * n_lset + q * u * n_lset) * dS

        if self.add_number_space:
            self.stokes_term += ((p * s + q * r) - (1e-8  * r * s)) * dx_neg
        else:
            self.stokes_term += 1e-10 * p * q * dx_neg

        P_gamma = Id(self.mesh.dim) - OuterProduct(n_lset, n_lset)
        P_S = Id(self.mesh.dim) - OuterProduct(n_bnd, n_bnd)
        div_gamma = lambda w: div(w) - InnerProduct(n_lset, grad(w) * n_lset)

        d_contact_plane = dCut(self.lset.lsetp1, domain_type=NEG,
                               deformation=self.lset.deformation, vb=BND,
                               definedon=self.mesh.Boundaries(self.wall_params.region))
        d_contact_line = dCut(self.lset.lsetp1, domain_type=IF,
                               deformation=self.lset.deformation, vb=BND,
                              definedon=self.mesh.Boundaries(self.wall_params.region))

        beta_S = self.wall_params.friction_coeff_surface
        beta_L = self.wall_params.friction_coeff_line
        theta_e = self.wall_params.contact_angle
        if self.mesh.dim == 2:
            t = specialcf.tangential(2)
            n_line = IfPos(InnerProduct(t, n_lset), t, -t)
        else:
            n_lset_surf = n_lset - InnerProduct(n_lset, n_bnd) * n_bnd
            n_line = n_lset_surf / (Norm(n_lset_surf) + 1e-12)

        self.stokes_term += 1/self.rho * beta_S * InnerProduct(P_S * u, P_S * v) * d_contact_plane
        self.stokes_term += 1/self.rho * beta_L * InnerProduct(u*n_line, v*n_line) * d_contact_line

    def AssembleConvection(self):
        trial, test = self.fes.TnT()
        u, p = trial[0], trial[1]
        v, q = test[0], test[1]

        dx_neg = self.lset.dx_neg
        u_approx = self.intermediate.components[0]

        self.conv = self.conv = (grad(u) * u_approx) * v * dx_neg + (grad(u_approx) * u) * v * dx_neg

        if self.use_supg:
            h = specialcf.mesh_size
            W = L2(self.mesh, order=0)
            gamma_gfu = GridFunction(W)
            gamma_gfu.Set(h / (2 * Norm(u_approx) + 1e-8))
            gamma_cf = CoefficientFunction(gamma_gfu)

            self.conv += gamma_cf * (InnerProduct(grad(u) * u_approx,  grad(v) * u_approx)) * dx_neg

    @timed_method
    def AssembleTimeStepping(self):
        trial, test = self.fes.TnT()
        u, p = trial[0], trial[1]
        v, q = test[0], test[1]

        dx_neg = self.lset.dx_neg

        self.mass = u * v * dx_neg
        self.mass_op = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer, facet_restriction=self.facets_ring, check_unused=False)
        self.mass_op += self.mass
        self.mass_op.Assemble(reallocate=True)
        # implicit factor of the effective scheme: the first validated step
        # runs backward Euler (full dt), afterwards BDF2 (startup consistency).
        beta = 1.0 if self.EffectiveTimeOrder() == 1 else 2.0 / 3.0
        self.m_star = RestrictedBilinearForm(self.fes, element_restriction=self.els_outer, facet_restriction=self.facets_ring, check_unused=False)
        self.m_star += self.mass + beta * self.dt * self.stokes_term
        if self.add_convection:
            self.m_star += beta * self.dt * self.conv
        self.m_star.Assemble(reallocate=True)
        self._assembled_beta = beta

    @timed_method
    def InvertTimeStepping(self):
        self.inv = self.m_star.mat.Inverse(freedofs=self.active_dofs & self.fes.FreeDofs(), inverse=direct_solver_nonspd)


    def SolveStokes(self):
        gfup = GridFunction(self.fes)
        gfu = gfup.components[0]
        default = CF((0,0)) if self.mesh.dim == 2 else CF((0,0,0))
        cf = self.mesh.BoundaryCF(self.boundary_registry.strong_dirichlet_dict, default=default)
        gfu.Set(cf, definedon=self.mesh.Boundaries(self.boundary_registry.dbnd))

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

    @timed_method
    def Step(self):
        # BDF scheme of the effective order (startup: step 1 = backward Euler).
        if self.EffectiveTimeOrder() == 1:   # startup: backward Euler
            weights, beta = (1.0,), 1.0
        else:                                # BDF2
            weights, beta = (4.0 / 3.0, -1.0 / 3.0), 2.0 / 3.0
        if beta != self._assembled_beta:
            self.AssembleTimeStepping()
            self.InvertTimeStepping()
        self.AssembleLf()
        history = (self.past, self.ancient)
        res = beta * self.dt * self.lf.vec - self.m_star.mat * self.gfup.vec
        for w_i, u_i in zip(weights, history):
            res += w_i * (self.mass_op.mat * u_i.vec)
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
