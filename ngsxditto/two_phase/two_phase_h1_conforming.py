from ngsolve import *
from xfem import *

from ngsxditto.levelset import LevelSetGeometry, DummyLevelSet
from ngsxditto import direct_solver_spd, direct_solver_nonspd
import ngsolve.webgui as ngw
from ngsxditto.fluid import *
from .two_phase_discretization import *

class TwoPhaseH1Conforming(TwoPhaseDiscretization):
    """
    This class handles all two-phase H1-conforming fluid discretizations.
    """
    def __init__(self, mesh: Mesh, fluid1_params: FluidParameters, fluid2_params: FluidParameters, order:int,
                 lset:LevelSetGeometry, wall_params: WallParameters, time_order:int,
                 f1: CoefficientFunction, f2: CoefficientFunction,  g1: CoefficientFunction,
                 g2: CoefficientFunction, add_convection:bool,
                 surface_tension: CoefficientFunction, dt:float, nitsche_stab:int, ghost_stab:int, extension_radius:float,
                 derivative_jumps:bool, add_number_space=bool):
        """
        Initializes the fluid discretization with the given parameters and levelset.
        Parameters:
        ----------
        mesh: Mesh
            The computational mesh
        fluid1_params: FluidParameters
            Parameters of the first fluid (corresponding to the negative part of the levelset.)
        fluid2_params: FluidParameters
            Parameters of the second fluid (corresponding to the negative part of the levelset.)
        order: int
            the polynomial order
        lset: LevelsetGeometry
            The levelset that characterizes the unfitted domain.
        wall_params: WallParameters
            wall parameters for contact problems
        f1: CoefficientFunction
            The force term of the first phase.
        f2: CoefficientFunction
            The force term of the second phase.
        g1: CoefficientFunction
            The divergence constraint of the first phase.
        g2: CoefficientFunction
            The divergence constraint of the second phase.
        surface_tension: CoefficientFunction
            The surface tension force.
        dt: float
            Time-step size
        nitsche_stab: int
            The stabilization parameter for the nitsche term
        ghost_stab: int
            The ghost stability parameter
        extension_radius: float
            Radius around the zero levelset on which the domain is extended.
        """
        super().__init__(mesh=mesh, fluid1_params=fluid1_params, fluid2_params=fluid2_params, order=order, lset=lset,
                         wall_params=wall_params, f1=f1, f2=f2, g1=g1, g2=g2, add_convection=add_convection,
                         surface_tension=surface_tension, dt=dt, time_order=time_order,
                         derivative_jumps=derivative_jumps, add_number_space=add_number_space)

        self.els_outer = None
        self.els_inner = None
        self.facets_ring = None

        self.ghost_stab = ghost_stab
        self.nitsche_stab = nitsche_stab    # nitsche stabilization
        self.extension_radius = extension_radius    # extension ring

        self.lsetp1_outer = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field - self.extension_radius, self.lsetp1_outer)

        self.lsetp1_inner = GridFunction(H1(self.mesh, order=1))
        InterpolateToP1(self.lset.field + self.extension_radius, self.lsetp1_inner)

        self.ci_main = CutInfo(self.mesh, self.lset.lsetp1)
        self.ci_inner = CutInfo(self.mesh, self.lsetp1_inner)
        self.ci_outer = CutInfo(self.mesh, self.lsetp1_outer)
        #els_hasneg = self.ci_main.GetElementsOfType(HASNEG)
        self.els_outer = self.ci_outer.GetElementsOfType(HASNEG)
        self.els_inner = self.ci_inner.GetElementsOfType(NEG)
        self.EA1 = ElementAggregation(mesh)
        self.EA2 = ElementAggregation(mesh)


    def SetInitialValues(self, initial_velocity1:CoefficientFunction, initial_velocity2:CoefficientFunction,
                         initial_pressure1:CoefficientFunction=CF(0), initial_pressure2:CoefficientFunction=CF(0),
                         mean_pressure_fix=None):
        self.mesh.SetDeformation(self.lset.deformation)
        self.gfu.components[0].Set(initial_velocity1)
        self.gfp.components[0].Set(initial_pressure1)
        self.gfu.components[1].Set(initial_velocity2)
        self.gfp.components[1].Set(initial_pressure2)
        self.mesh.UnsetDeformation()

        self.ValidateStep()


    def UpdateActiveDofs(self):
        """
        Updates the dofs that are active, i.e. all dofs that are in the extended unfitted domain.
        """
        InterpolateToP1(self.lset.field - self.extension_radius, self.lsetp1_outer)

        InterpolateToP1(self.lset.field + self.extension_radius, self.lsetp1_inner)

        self.ci_main.Update(self.lset.lsetp1)
        self.ci_inner.Update(self.lsetp1_inner)
        self.ci_outer.Update(self.lsetp1_outer)

        # Element and facet markers
        els_hasneg = self.ci_main.GetElementsOfType(HASNEG)
        els_haspos = self.ci_main.GetElementsOfType(HASPOS)
        self.els_outer = self.ci_outer.GetElementsOfType(HASNEG)
        self.els_inner = self.ci_inner.GetElementsOfType(NEG)
        els_ring = self.els_outer & ~self.els_inner
        self.facets_ring = GetFacetsWithNeighborTypes(self.mesh, a=self.els_outer, b=els_ring)
        self.EA1.Update(els_hasneg & ~self.lset.hasif, self.lset.hasif | (self.els_outer & ~ els_hasneg))
        self.EA2.Update(els_haspos & ~self.lset.hasif, self.lset.hasif | (~self.els_inner & ~ els_haspos))

    def InitializeForms(self):
        self.AssembleLf()

        if self.add_convection:
            self.AssembleConvection()

        self.AssembleStokes()

        self.AssembleTimeStepping()
        self.InvertTimeStepping()


    def AssembleLf(self):
        test = self.fes.TestFunction()
        v, q = test[0], test[1]

        nus = [self.nu1, self.nu2]
        dx_neg = self.lset.dx_neg
        dx_pos = self.lset.dx_pos
        dx_list = [dx_neg, dx_pos]
        f_list = [self.f1, self.f2]
        g_list = [self.g1, self.g2]
        dS = self.lset.dS
        kappaminus = CutRatioGF(self.lset.cutinfo)
        kappa = (kappaminus, 1 - kappaminus)
        h = specialcf.mesh_size
        n_bnd = specialcf.normal(self.mesh.dim)
        n_lset = self.lset.n

        surface_tension_list = [self.fluid1_params.surface_tension_coeff, self.fluid2_params.surface_tension_coeff]
        self.lf = LinearForm(self.fes)
        for i in range(2):
            self.lf += f_list[i] * v[i] * dx_list[i]
            self.lf += g_list[i] * q[i] * dx_list[i]
            for (region, values) in self.boundary_registry.nitsche_normal_velocity_dict.items():
                if region != "interface":
                    self.lf += (-nus[i] * (grad(v[i]).Trace() * n_bnd) * n_bnd * values
                                + nus[i] * self.nitsche_stab / h * (v[i] * n_bnd) * values) * ds(
                        definedon=self.mesh.Boundaries(region))
                else:
                    self.lf += (-nus[i] * (grad(v[i]).Trace() * n_lset) * n_lset * values
                                + nus[i] * self.nitsche_stab / h * (v[i] * n_lset) * values) * dS

            for (region, values) in self.boundary_registry.nitsche_velocity_dict.items():
                if region != "interface":
                    self.lf += (-nus[i] * grad(v[i]) * n_bnd * values +
                                nus[i] * self.nitsche_stab / h * values * v[i] +
                                q[i] * n_bnd * values) * ds(definedon=self.mesh.Boundaries(region))
                else:
                    self.lf += (-nus[i] * grad(v[i]) * n_lset * values +
                                nus[i] * self.nitsche_stab / h * values * v[i] +
                                q[i] * n_lset * values) * dS

            for (region, values) in self.boundary_registry.strong_neumann_dict.items():
                self.lf += nus[i] * values * v[i] * dx(definedon=self.mesh.Boundaries(region))

        if self.surface_tension is not None:
            self.lf += -surface_tension_list[0] * self.surface_tension * (kappa[1] * v[0] + kappa[0] * v[1]) * dS

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
        nus = [self.nu1, self.nu2]
        rhos = [self.rho1, self.rho2]
        dx_neg = self.lset.dx_neg
        dx_pos = self.lset.dx_pos
        dx_list = [dx_neg, dx_pos]

        dw_neg = dFacetPatch(definedonelements=self.EA1.patch_interior_facets, deformation=self.lset.deformation)
        dw_pos = dFacetPatch(definedonelements=self.EA2.patch_interior_facets, deformation=self.lset.deformation)
        dw_list = [dw_neg, dw_pos]
        dS = self.lset.dS
        n_F = specialcf.normal(self.mesh.dim)

        kappaminus = CutRatioGF(self.lset.cutinfo)
        kappa = (kappaminus, 1 - kappaminus)

        self.stokes_term = 0
        for i in range(2):
            basic_stokes = (nus[i] * InnerProduct(grad(u[i]), grad(v[i])) - 1/rhos[i] * p[i] * div(v[i]) - 1/rhos[i] * q[i] * div(u[i])) * dx_list[i]
            if not self.derivative_jumps:
                #dw = dFacetPatch(definedonelements=self.facets_ring, deformation=self.lset.deformation)
                ghost_u = 1/h**2 * (u[i] - u[i].Other()) * (v[i] - v[i].Other()) * dw_list[i]
                ghost_p = (p[i] - p[i].Other()) * (q[i] - q[i].Other()) * dw_list[i]
            else:
                dw = dx(skeleton=True, definedonelements=self.facets_ring, deformation=self.lset.deformation)
                ghost_p = h ** 3 * InnerProduct((grad(p[i]) - grad(p[i].Other())) * n_F,
                                                (grad(q[i]) - grad(q[i].Other())) * n_F) * dw

                ghost_u = h * InnerProduct((grad(u[i]) - grad(u[i].Other())) * n_F, (grad(v[i]) - grad(v[i].Other())) * n_F) * dw
                for j in range(self.mesh.dim):
                    ghost_u += h ** 3 * InnerProduct(
                        (u[i].Operator("hesse")[j] - u[i].Other().Operator("hesse")[j]) * n_F,
                        (v[i].Operator("hesse")[j] - v[i].Other().Operator("hesse")[j]) * n_F) * dw

            ghost_penalty = (self.ghost_stab * ghost_u - self.ghost_stab * ghost_p)

            for (region, values) in self.boundary_registry.nitsche_normal_velocity_dict.items():
                if region != "interface":
                    un = u[i] * n_bnd
                    vn = v[i] * n_bnd

                    nitsche = (-(grad(u[i]).Trace() * n_bnd) * n_bnd * vn - (grad(v[i]).Trace() * n_bnd) * n_bnd * un
                               + self.nitsche_stab / h * un * vn) * ds(definedon=self.mesh.Boundaries(region))
                    self.stokes_term += nus[i] * nitsche
                    self.stokes_term += (q[i] * u[i] * n_bnd + p[i] * v[i] * n_bnd) * ds(definedon=self.mesh.Boundaries(region))
                else:
                    un = u[i] * n_lset
                    vn = v[i] * n_lset

                    nitsche = (-(grad(u[i]).Trace() * n_lset) * n_lset * vn - (grad(v[i]).Trace() * n_lset) * n_lset * un
                               + self.nitsche_stab / h * un * vn) * dS
                    self.stokes_term += nus[i] * nitsche
                    self.stokes_term += (q[i] * u[i] * n_lset + p[i] * v[i] * n_lset) * dS

            for (region, values) in self.boundary_registry.nitsche_velocity_dict.items():
                if region != "interface":
                    nitsche = (-grad(u[i]).Trace() * n_bnd * v[i] - grad(v[i]).Trace() * n_bnd * u[i] +
                               self.nitsche_stab / h * u[i] * v[i]) * ds(definedon=self.mesh.Boundaries(region))
                    self.stokes_term += nus[i] * nitsche
                    self.stokes_term += (p[i] * v[i] * n_bnd + q[i] * u[i] * n_bnd) * ds(definedon=self.mesh.Boundaries(region))

                else:
                    nitsche = (-grad(u[i]) * n_lset * v[i] - grad(v[i]) * n_lset * u[i] + self.nitsche_stab / h * u[i] * v[i]) * dS
                    self.stokes_term += nus[i] * nitsche
                    self.stokes_term += (p[i] * v[i] * n_lset + q[i] * u[i] * n_lset) * dS

            if self.add_number_space:
                pressure_stab = (r[i] * q[i] + s[i] * p[i]) * dx_list[i]
            else:
                pressure_stab = 1e-10 * p[i] * q[i] * dx_list[i]
            self.stokes_term += basic_stokes + ghost_penalty + pressure_stab

        nitsche = (-(kappa[0]*nus[0]*grad(u[0]) * n_lset + kappa[1] * nus[1]*grad(u[1]) * n_lset) * (v[0] - v[1]) -
                   (kappa[0]*nus[0]*grad(v[0]) * n_lset + kappa[1] * nus[1]*grad(v[1]) * n_lset) * (u[0] - u[1]) +
                   self.nitsche_stab * (kappa[0] * nus[0] + kappa[1] * nus[1]) / h * (u[0] - u[1]) * (v[0] - v[1])) * dS

        bnd_terms = (((1/rhos[0] * kappa[0] * p[0] + 1/rhos[1] * kappa[1] * p[1]) * (v[0] - v[1]) * n_lset) * dS +
                     ((1/rhos[0] * kappa[0] * q[0] + 1/rhos[1] * kappa[1] * q[1]) * (u[0] - u[1]) * n_lset) * dS)
        self.stokes_term += nitsche + bnd_terms

        self.stokes_op = BilinearForm(self.fes)
        self.stokes_op += self.stokes_term
        self.stokes_op.Assemble()

    @timed_method
    def AssembleConvection(self):
        trial, test = self.fes.TnT()
        u, p = trial[0], trial[1]
        v, q = test[0], test[1]
        dx_neg = self.lset.dx_neg
        dx_pos = self.lset.dx_pos
        dx_list = [dx_neg, dx_pos]
        rhos = [self.rho1, self.rho2]

        self.conv = 0
        for i in range(2):
            self.conv += 1/rhos[i]*(grad(u[i]) * self.intermediate.components[0].components[i]) * v[i] * dx_list[i]
            self.conv += 1/rhos[i]*(grad(self.intermediate.components[0].components[i]) * u[i]) * v[i] * dx_list[i]
            self.conv -= 1/rhos[i]*(grad(self.intermediate.components[0].components[i]) * self.intermediate.components[0].components[i]) * v[i] * dx_list[i]

        self.conv_op = BilinearForm(self.fes)
        self.conv_op += self.conv
        self.conv_op.Assemble()

    @timed_method
    def AssembleTimeStepping(self):
        trial, test = self.fes.TnT()
        u, p = trial[0], trial[1]
        v, q = test[0], test[1]

        rhos = [self.rho1, self.rho2]
        dx_neg = self.lset.dx_neg
        dx_pos = self.lset.dx_pos
        dx_list = [dx_neg, dx_pos]

        mass1 = u[0] * v[0] * dx_list[0]
        mass2 = u[1] * v[1] * dx_list[1]
        mass_list = [mass1, mass2]

        self.mass_op = BilinearForm(self.fes)
        self.mass_op += mass1 + mass2
        self.mass_op.Assemble()
        coef = 2/3 if self.time_order == 2 else 1
        self.m_star = BilinearForm(self.fes)

        for i in range(2):
            self.m_star += mass_list[i]

        self.m_star += coef * self.dt * self.stokes_term
        if self.add_convection:
            self.m_star += coef * self.dt * self.conv

        self.m_star.Assemble()

    @timed_method
    def InvertTimeStepping(self):
        self.inv = self.m_star.mat.DeleteZeroElements(1e-100).Inverse(freedofs=self.fes.FreeDofs(), inverse=direct_solver_nonspd)
        #self.inv = self.m_star.mat.Inverse(freedofs=self.fes.FreeDofs(), inverse=direct_solver_nonspd)


    def SolveStokes(self):
        gfup = GridFunction(self.fes)
        gfu, gfp = gfup.components[0], gfup.components[1]
        default = CF((0,0)) if self.mesh.dim == 2 else CF((0,0,0))
        cf = self.mesh.BoundaryCF(self.boundary_registry.strong_dirichlet_dict, default=default)
        gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.boundary_registry.dbnd))
        gfu.components[1].Set(cf, definedon=self.mesh.Boundaries(self.boundary_registry.dbnd))

        gfup.vec.data += self.stokes_op.mat.DeleteZeroElements(1e-100).Inverse(self.fes.FreeDofs(), inverse=direct_solver_nonspd) * (self.lf.vec - self.stokes_op.mat * gfup.vec)

        return gfup

    @timed_method
    def Step(self):
        if self.time is not None:
            self.time += self.dt

        self.AssembleLf()

        if self.time_order == 1:
            res = self.mass_op.mat * self.past.vec + self.dt * self.lf.vec - self.m_star.mat * self.gfup.vec
            self.gfup.vec.data += self.inv * res

        elif self.time_order >= 2:
            res = (4/3) * self.mass_op.mat * self.past.vec \
                  - (1/3) * self.mass_op.mat * self.ancient.vec \
                  + (2/3) * self.dt * self.lf.vec \
                  - self.m_star.mat * self.gfup.vec
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
        self.AssembleTimeStepping()
        self.InvertTimeStepping()
