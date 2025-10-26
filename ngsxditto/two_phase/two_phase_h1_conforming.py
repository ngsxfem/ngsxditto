from ngsolve import *
from xfem import *

from ngsxditto.levelset import LevelSetGeometry, DummyLevelSet
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
                 g2: CoefficientFunction = CF(0),surface_tension: CoefficientFunction = None, dt=None,
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
                         wall_params=wall_params, f1=f1, f2=f2, g1=g1, g2=g2,
                         surface_tension=surface_tension, dt=dt, if_dirichlet=if_dirichlet)
        self.V_base = None
        self.Q_base = None

        self.els_outer = None
        self.els_inner = None
        self.facets_ring = None
        self.active_u_dofs_1 = None
        self.active_u_dofs_2 = None
        self.active_p_dofs_1 = None
        self.active_p_dofs_2 = None

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



    def SetInitialValues(self, initial_velocity1:CoefficientFunction, initial_velocity2:CoefficientFunction,
                         initial_pressure1:CoefficientFunction=CF(0), initial_pressure2:CoefficientFunction=CF(0)):
        self.gfu.components[0].Set(initial_velocity1)
        self.gfu.components[1].Set(initial_pressure1)
        self.gfu.components[2].Set(initial_velocity2)
        self.gfu.components[3].Set(initial_pressure2)

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
        self.els_inner = self.ci_inner.GetElementsOfType(NEG)
        els_ring = self.els_outer & ~self.els_inner
        self.facets_ring = GetFacetsWithNeighborTypes(self.mesh, a=self.els_outer, b=els_ring)
        self.active_u_dofs_1 = GetDofsOfElements(self.V_base, self.els_outer)
        self.active_u_dofs_2 = GetDofsOfElements(self.V_base, ~self.els_inner)
        self.active_p_dofs_1 = GetDofsOfElements(self.Q_base, self.els_outer)
        self.active_p_dofs_2 = GetDofsOfElements(self.Q_base, ~self.els_inner)



    def InitializeForms(self):
        u1, p1, u2, p2 = self.fes.TrialFunction()
        v1, q1, v2, q2 = self.fes.TestFunction()
        u = [u1, u2]
        v = [v1, v2]
        p = [p1, p2]
        q = [q1, q2]
        h = specialcf.mesh_size
        n = self.lset.n
        rhos = [self.rho1, self.rho2]
        nus = [self.nu1, self.nu2]
        dx_neg = self.lset.dx_neg
        dx_pos = self.lset.dx_pos
        dx_list = [dx_neg, dx_pos]
        f_list = [self.f1, self.f2]
        g_list = [self.g1, self.g2]
        dS = self.lset.dS
        mass1 = u[0] * v[0] * dx_list[0]
        mass2 = u[1] * v[1] * dx_list[1]
        mass_list = [mass1, mass2]

        surface_tension_list = [self.fluid1_params.surface_tension_coeff, self.fluid2_params.surface_tension_coeff]

        kappaminus = CutRatioGF(self.lset.cutinfo)
        kappa = (kappaminus, 1 - kappaminus)

        self.lf = LinearForm(self.fes)
        for i in range(2):
            self.lf += rhos[i] * f_list[i] * v[i] * dx_list[i]
            self.lf += g_list[i] * q[i] * dx_list[i]

            for (region, fct) in self.neumann.items():
                self.lf += nus[i] * fct * v[i] * dx(definedon=self.mesh.Boundaries(region))

        if self.surface_tension is not None:
            self.lf += -surface_tension_list[0] * self.surface_tension * (kappa[0] * v[0] + kappa[1] * v[1]) * dS

        self.lf.Assemble()


        self.a = BilinearForm(self.fes)
        dw = dFacetPatch(definedonelements=self.facets_ring, deformation=self.lset.deformation)
        stokes_list = []
        for i in range(2):
            basic_stokes = (nus[i] * InnerProduct(grad(u[i]), grad(v[i])) - p[i] * div(v[i]) - q[i] * div(u[i])) * dx_list[i]

            ghost_u = 1/h**2 * (u[i] - u[i].Other()) * (v[i] - v[i].Other()) * dw
            ghost_p = (p[i] - p[i].Other()) * (q[i] - q[i].Other()) * dw
            ghost_penalty = self.ghost_stab * nus[i] * ghost_u - self.ghost_stab * 1/nus[i] * ghost_p + self.ghost_stab * 1/nus[i] * ghost_u

            stokes = basic_stokes + ghost_penalty
            stokes_list.append(stokes)
            self.a += stokes


        nitsche = (-(kappa[0]*nus[0]*grad(u[0]) * n + kappa[1] * nus[1]*grad(u[1]) * n) * (v[0] - v[1]) -
                   (kappa[0]*nus[0]*grad(v[0]) * n + kappa[1] * nus[1]*grad(v[1]) * n) * (u[0] - u[1]) +
                   self.nitsche_stab / h * (u[0] - u[1]) * (v[0] - v[1])) * dS

        bnd_terms = (((kappa[0] * p[0] + kappa[1] * p[1]) * (v[0] - v[1]) * n) * dS +
                     ((kappa[0] * q[0] + kappa[1] * q[1]) * (u[0] - u[1]) * n) * dS)

        self.a += nitsche
        self.a += bnd_terms

        self.a.Assemble()

        self.m_star = BilinearForm(self.fes)

        for i in range(2):
            self.m_star += rhos[i] * mass_list[i]
            self.m_star += self.dt * stokes_list[i]
        self.m_star += nitsche
        self.m_star += bnd_terms
        self.m_star.Assemble()

        freedofs = self.fes.FreeDofs()
        #freedofs &= CompoundBitArray([self.active_u_dofs_1, self.active_p_dofs_1,
        #                              self.active_u_dofs_2, self.active_p_dofs_2])

        self.inv = self.m_star.mat.Inverse(freedofs=freedofs)



    def SolveStokes(self):
        gfu = GridFunction(self.fes)
        default = CF((0,0)) if self.mesh.dim == 2 else CF((0,0,0))
        cf = self.mesh.BoundaryCF(self.dirichlet, default=default)
        gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))
        gfu.components[2].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))

        gfu.vec.data += self.a.mat.Inverse(self.fes.FreeDofs()) * (self.lf.vec - self.a.mat * gfu.vec)

        return gfu


    def Step(self):
        if self.time is not None:
            self.time += self.dt

        self.InitializeForms()
        res = self.lf.vec - self.a.mat * self.gfu.vec
        self.gfu.vec.data += self.dt * self.inv * res


    def SetTimeStepSize(self, dt):
        raise(NotImplementedError("Not yet Implemented"))
