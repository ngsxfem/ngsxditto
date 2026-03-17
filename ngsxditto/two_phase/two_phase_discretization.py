from ngsxditto.fluid import *
from ngsxditto.stepper import *
from ngsolve import *
import typing

class TwoPhaseDiscretization(GFStepper):
    """
    Base class for two-phase fluid discretizations.
    """
    def __init__(self, mesh: Mesh, fluid1_params: FluidParameters, fluid2_params: FluidParameters, dt:float, order: int,
                 lset:LevelSetGeometry, if_dirichlet:CoefficientFunction, wall_params: WallParameters, add_convection:bool,
                 f1:CoefficientFunction, f2: CoefficientFunction, g1: CoefficientFunction, g2: CoefficientFunction,
                 surface_tension:CoefficientFunction, derivative_jumps:bool, add_number_space:bool,
                 time: typing.Optional[Parameter] = None):
        """
        Creates a two-phase fluid discretization on the given mesh defined by the levelset.
        If no levelset is given, create a DummyLevelSet that covers the whole domain.

        Parameters:
        -----------
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
        if_dirichlet: CoefficientFunction
            Dirichlet boundary condition of the unfitted domain.
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
        time: Parameter
            The time parameter.
        """

        super().__init__()
        self.mesh = mesh
        self.fluid1_params = fluid1_params
        self.fluid2_params = fluid2_params
        self.order = order
        if lset is None:
            self.lset = DummyLevelSet(mesh)
        else:
            self.SetLevelSet(lset)

        self.add_convection = add_convection

        self.if_dirichlet = if_dirichlet
        self.wall_params = wall_params
        default = CF((0, 0)) if self.mesh.dim == 2 else CF((0, 0, 0))
        if f1 is None:
            self.f1 = default
        else:
            self.f1 = f1
        if f2 is None:
            self.f2 = default
        else:
            self.f2 = f2
        self.g1 = g1
        self.g2 = g2
        self.derivative_jumps = derivative_jumps
        self.add_number_space = add_number_space
        if surface_tension is None:
            self.surface_tension = default
        else:
            self.surface_tension = surface_tension
        self.dt = dt
        self.time = time
        self.gfup = None
        self.gfu = None
        self.gfp = None
        self.gfn = None
        self.stokes_op = None
        self.lf = None
        self.conv = None
        self.m_star = None
        self.inv = None
        self.mass = None
        self.stokes = None
        self.fes = None
        self.dirichlet = None
        self.neumann = None
        self.dbnd = None
        self.dt = dt
        self.nu1 = self.fluid1_params["viscosity"]
        self.nu2 = self.fluid2_params["viscosity"]
        self.rho1 = self.fluid1_params["density"]
        self.rho2 = self.fluid2_params["density"]
        self.time = time
        self.multistepper = MultiStepper()
        self.multistepper.SetObject(self)


    def Initialize(self, dirichlet:dict=None, neumann:dict=None,
                   initial_velocity1:CoefficientFunction=None,
                   initial_velocity2: CoefficientFunction = None,
                   initial_pressure1:CoefficientFunction=CF(0),
                   initial_pressure2: CoefficientFunction = CF(0),
                   ):
        """
        Initializes the fluid discretization, setting boundary conditions of the outer as well as
        physical domain and initializing the finite element spaces and bilinear forms.
        Convenience function that combines SetBoundaryConditions, InitializeSpaces,
        ApplyBoundaryConditions, UpdateActiveDofs and InitializeForms.

        Parameters:
        -----------
        dirichlet: dict
            A dictionary with dirichlet boundary conditions of the form
            {"region (str)": function (CoefficientFunction), ...}
        neumann: dict
            A dictionary with neumann boundary conditions of the form
            {"region (str)": function (CoefficientFunction), ...}
        initial_velocity1: CoefficientFunction
            The initial velocity of the fluid in \Omega^{-}.
        initial_velocity2: CoefficientFunction
            The initial velocity of the fluid in \Omega^{+}.
        initial_pressure1: CoefficientFunction
            The initial pressure of the fluid in \Omega^{-}.
        initial_pressure2: CoefficientFunction
            The initial pressure of the fluid in \Omega^{+}.

        """
        default = CF((0, 0)) if self.mesh.dim == 2 else CF((0, 0, 0))
        if initial_velocity1 is None:
            initial_velocity1 = default
        if initial_velocity2 is None:
            initial_velocity2 = default

        self.SetBoundaryConditions(dirichlet=dirichlet, neumann=neumann)
        self.InitializeBaseSpaces()
        self.UpdateActiveDofs()
        self.InitializeCombinedSpace()
        self.InitializeGfu()
        self.ApplyBoundaryConditions()
        self.lset.lsetadap.ProjectOnUpdate([self.current.components[i].components[j] for i in range(2) for j in range(2)] +
                                           [self.intermediate.components[i].components[j] for i in range(2) for j in range(2)] +
                                           [self.past.components[i].components[j] for i in range(2) for j in range(2)])

        self.InitializeForms()
        self.SetInitialValues(initial_velocity1, initial_velocity2, initial_pressure1, initial_pressure2)


    def SetBoundaryConditions(self, dirichlet:dict=None, neumann:dict=None):
        """
        Set the dirichlet and neumann boundary conditions for your problem.

        Parameters:
        -----------
        dirichlet: dict
            A dictionary with dirichlet boundary conditions of the form
            {"region (str)": function (CoefficientFunction), ...}
        neumann: dict
            A dictionary with neumann boundary conditions of the form
            {"region (str)": function (CoefficientFunction), ...}
        """
        if dirichlet is None:
            dirichlet = {}

        if neumann is None:
            neumann = {}

        self.dirichlet = dirichlet
        self.neumann = neumann
        self.dbnd = "|".join(dirichlet.keys())


    def SetInitialValues(self, initial_velocity1:CoefficientFunction, initial_velocity2:CoefficientFunction,
                         initial_pressure1:CoefficientFunction=CF(0), initial_pressure2:CoefficientFunction=CF(0),
                         mean_pressure_fix=None):
        """
        Sets the initial values for velocity and pressure
        """
        raise NotImplementedError("SetInitialValues not implemented.")


    def ApplyBoundaryConditions(self):
        """
        Applies the boundary conditions after they are set with SetBoundaryConditions and after the spaces
        are defined with InitializeSpaces.
        """
        cf = self.mesh.BoundaryCF(self.dirichlet)
        self.gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))
        self.gfu.components[1].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))


    def InitializeSpaces(self):
        """
        Initializes the Finite element spaces.
        """
        raise NotImplementedError("InitializeSpaces not implemented.")


    def UpdateActiveDofs(self):
        """
        Updates the active degrees of freedom after a levelset update.
        """
        raise NotImplementedError("UpdateActiveDofs not implemented.")


    def InitializeForms(self):
        """
        Initializes the bilinear and linear forms.
        """
        raise NotImplementedError("InitializeForms not implemented.")


    def SetLevelSet(self, lset:LevelSetGeometry):
        """
        Sets the levelset that describes the unfitted domain.
        """
        self.lset = lset
        if self.UpdateActiveDofs not in lset.callbacks:
            self.lset.callbacks.append(self.UpdateActiveDofs)
        if self.InitializeForms not in lset.callbacks:
            self.lset.callbacks.append(self.InitializeForms)

    def SetTimeStepSize(self, dt):
        """
        Sets the time step size and reassembles the necessary forms.
        """
        raise NotImplementedError("SetTimeStepSize not implemented.")


    def ComputeDifference2Intermediate(self):
        """
        return difference in velocity L2(Omega_tilde) norm where
        Omega_tilde is the background mesh
        """
        inner_integal = Integrate((self.current.components[0].components[0] -
                                   self.intermediate.components[0].components[0])**2 * self.lset.dx_neg,self.mesh)
        outer_integral = Integrate((self.current.components[0].components[1] -
                                   self.intermediate.components[0].components[1])**2 * self.lset.dx_pos,self.mesh)
        return (inner_integal + outer_integral)**(1/2)

    def Step(self):
        raise NotImplementedError("Step only implemented in subclasses.")
