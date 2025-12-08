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
    def __init__(self, mesh, fluid1_params: FluidParameters, fluid2_params: FluidParameters, order=4,
                 lset:LevelSetGeometry=None,wall_params: WallParameters = None, if_dirichlet:CoefficientFunction=None,
                 f1: CoefficientFunction = None, f2: CoefficientFunction = None,  g1: CoefficientFunction = CF(0),
                 g2: CoefficientFunction = CF(0), add_convection:bool=False, fix_point_eps:float = 1e-2,
                 surface_tension: CoefficientFunction = None, dt=None,
                 nitsche_stab:int=100, ghost_stab:int=20, extension_radius:float=0.2):
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
        if_dirichlet: CoefficientFunction
            Dirichlet boundary condition of the unfitted domain.
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
                         fix_point_eps=fix_point_eps, surface_tension=surface_tension, dt=dt, if_dirichlet=if_dirichlet)

        self.els_outer = None
        self.els_inner = None
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
        #els_hasneg = self.ci_main.GetElementsOfType(HASNEG)
        self.els_outer = self.ci_outer.GetElementsOfType(HASNEG)
        self.els_inner = self.ci_inner.GetElementsOfType(NEG)



    def SetInitialValues(self, initial_velocity1:CoefficientFunction, initial_velocity2:CoefficientFunction,
                         initial_pressure1:CoefficientFunction=CF(0), initial_pressure2:CoefficientFunction=CF(0),
                         mean_pressure_fix=None):
        self.gfu.components[0].Set(initial_velocity1)
        self.gfp.components[0].Set(initial_pressure1)
        self.gfu.components[1].Set(initial_velocity2)
        self.gfp.components[1].Set(initial_pressure2)

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
        els_ring = self.els_outer & ~self.els_inner
        self.facets_ring = GetFacetsWithNeighborTypes(self.mesh, a=self.els_outer, b=els_ring)


    def InitializeForms(self):
        self.AssembleLf()

        if self.add_convection:
            self.AssembleConvection()

        self.AssembleStokes()

        self.AssembleInvertTimeStepping()


    def AssembleLf(self):
        v, q, s = self.fes.TestFunction()

        rhos = [self.rho1, self.rho2]
        nus = [self.nu1, self.nu2]
        dx_neg = self.lset.dx_neg
        dx_pos = self.lset.dx_pos
        dx_list = [dx_neg, dx_pos]
        f_list = [self.f1, self.f2]
        g_list = [self.g1, self.g2]
        dS = self.lset.dS
        kappaminus = CutRatioGF(self.lset.cutinfo)
        kappa = (kappaminus, 1 - kappaminus)

        surface_tension_list = [self.fluid1_params.surface_tension_coeff, self.fluid2_params.surface_tension_coeff]

        self.lf = LinearForm(self.fes)
        for i in range(2):
            self.lf += rhos[i] * f_list[i] * v[i] * dx_list[i]
            self.lf += g_list[i] * q[i] * dx_list[i]

            for (region, fct) in self.neumann.items():
                self.lf += nus[i] * fct * v[i] * dx(definedon=self.mesh.Boundaries(region))

        if self.surface_tension is not None:
            self.lf += -surface_tension_list[0] * self.surface_tension * (kappa[0] * v[0] + kappa[1] * v[1]) * dS

        self.lf.Assemble()


    def AssembleStokes(self):
        u, p, r = self.fes.TrialFunction()
        v, q, s = self.fes.TestFunction()

        h = specialcf.mesh_size
        n = self.lset.n
        nus = [self.nu1, self.nu2]
        dx_neg = self.lset.dx_neg
        dx_pos = self.lset.dx_pos
        dx_list = [dx_neg, dx_pos]
        dS = self.lset.dS


        kappaminus = CutRatioGF(self.lset.cutinfo)
        kappa = (kappaminus, 1 - kappaminus)
        dw = dFacetPatch(definedonelements=self.facets_ring, deformation=self.lset.deformation)

        self.stokes_term = 0
        for i in range(2):
            basic_stokes = (nus[i] * InnerProduct(grad(u[i]), grad(v[i])) - p[i] * div(v[i]) - q[i] * div(u[i])) * dx_list[i]

            ghost_u = 1/h**2 * (u[i] - u[i].Other()) * (v[i] - v[i].Other()) * dw
            ghost_p = (p[i] - p[i].Other()) * (q[i] - q[i].Other()) * dw
            ghost_penalty = self.ghost_stab * ghost_u - self.ghost_stab * ghost_p
            #ghost_penalty = (self.ghost_stab * nus[i] * ghost_u - self.ghost_stab * 1 / nus[i] * ghost_p +
            #                 self.ghost_stab * 1 / nus[i] * ghost_u)

            pressure_stab = (r * q[i] + s * p[i]) * dx_list[i]
            self.stokes_term += basic_stokes + ghost_penalty #+ pressure_stab

        nitsche = (-(kappa[0]*nus[0]*grad(u[0]) * n + kappa[1] * nus[1]*grad(u[1]) * n) * (v[0] - v[1]) -
                   (kappa[0]*nus[0]*grad(v[0]) * n + kappa[1] * nus[1]*grad(v[1]) * n) * (u[0] - u[1]) +
                   self.nitsche_stab / h * (u[0] - u[1]) * (v[0] - v[1])) * dS

        bnd_terms = (((kappa[0] * p[0] + kappa[1] * p[1]) * (v[0] - v[1]) * n) * dS +
                     ((kappa[0] * q[0] + kappa[1] * q[1]) * (u[0] - u[1]) * n) * dS)

        self.stokes_term += nitsche + bnd_terms

        self.stokes_op = BilinearForm(self.fes)
        self.stokes_op += self.stokes_term
        self.stokes_op.Assemble()


    def AssembleConvection(self):
        u, p, r = self.fes.TrialFunction()
        v, q, s = self.fes.TestFunction()
        dx_neg = self.lset.dx_neg
        dx_pos = self.lset.dx_pos
        dx_list = [dx_neg, dx_pos]

        self.conv = 0
        for i in range(2):
            self.conv += (grad(u[i]) * self.intermediate.components[0].components[i]) * v[i] * dx_list[i]

        self.conv_op = BilinearForm(self.fes)
        self.conv_op += self.conv
        self.conv_op.Assemble()


    def AssembleInvertTimeStepping(self):
        u, p, r = self.fes.TrialFunction()
        v, q, s = self.fes.TestFunction()

        rhos = [self.rho1, self.rho2]
        dx_neg = self.lset.dx_neg
        dx_pos = self.lset.dx_pos
        dx_list = [dx_neg, dx_pos]

        mass1 = u[0] * v[0] * dx_list[0]
        mass2 = u[1] * v[1] * dx_list[1]
        mass_list = [mass1, mass2]

        self.m_star = BilinearForm(self.fes)

        for i in range(2):
            self.m_star += rhos[i] * mass_list[i]

        self.m_star += self.dt * self.stokes_term
        if self.add_convection:
            self.m_star += self.dt * self.conv

        self.m_star.Assemble()

        freedofs = self.fes.FreeDofs()
        #freedofs &= CompoundBitArray([self.active_u_dofs_1, self.active_p_dofs_1,
        #                              self.active_u_dofs_2, self.active_p_dofs_2])
        self.inv = self.m_star.mat.Inverse(freedofs=freedofs, inverse=direct_solver_nonspd)



    def SolveStokes(self):
        gfup = GridFunction(self.fes)
        gfu, gfp, gfn = gfup.components
        default = CF((0,0)) if self.mesh.dim == 2 else CF((0,0,0))
        cf = self.mesh.BoundaryCF(self.dirichlet, default=default)
        gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))
        gfu.components[1].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))

        gfup.vec.data += self.stokes_op.mat.Inverse(self.fes.FreeDofs(), inverse=direct_solver_nonspd) * (self.lf.vec - self.stokes_op.mat * gfup.vec)

        return gfup


    def Step(self):
        if self.time is not None:
            self.time += self.dt

        self.AssembleLf()

        if not self.add_convection:
            res = self.lf.vec - self.stokes_op.mat * self.gfup.vec
            self.gfup.vec.data += self.dt * self.inv * res
        else:
            count = 0
            while True:
                count += 1
                self.AssembleConvection()
                self.AssembleInvertTimeStepping()
                res = self.lf.vec - self.stokes_op.mat * self.gfup.vec - self.conv_op.mat * self.gfup.vec
                self.gfup.vec.data += self.dt * self.inv * res
                diff_to_intermediate = self.ComputeDifference2Intermediate()
                if diff_to_intermediate <= self.fix_point_eps:
                    break
                self.RevertStep()

                if count >= 20:
                    print("Fix point iteration did not converge after 20 iterations. Difference to previous step was", diff_to_intermediate)
                    break


    def SetTimeStepSize(self, dt):
        self.dt = dt
        self.AssembleInvertTimeStepping()
