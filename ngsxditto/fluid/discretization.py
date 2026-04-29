"""
This file handles fluid discretizations.
"""
from .params import FluidParameters, WallParameters
from ngsxditto.boundary_registry import *
from ngsxditto.levelset import *
from ngsxditto.multistepper import MultiStepper
from ngsxditto.stepper import *
from xfem import *
import typing


class FluidDiscretization(GFStepper):
    """
    Base class for a discretized fluid.
    """
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int, lset:LevelSetGeometry,
                 wall_params: WallParameters, add_convection:bool,
                 f:CoefficientFunction, g: CoefficientFunction, surface_tension:CoefficientFunction, dt:float,
                 derivative_jumps:bool, add_number_space:bool, time_order:int, use_supg:bool,
                 time: typing.Optional[Parameter]=None
                 ):
        """
        Creates a fluid discretization on the given mesh under consideration of the levelset.
        If None is given, create a DummyLevelSet that covers the whole domain.

        Parameters:
        -----------
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
        f: CoefficientFunction
            The force term
        g: CoefficientFunction
            The divergence constraint
        surface_tension: CoefficientFunction
            The surface tension force.
        dt: float
            Time-step size
        time: Parameter
            The time parameter.
        """
        super().__init__()
        self.mesh = mesh
        self.fluid_params = fluid_params
        self.order = order
        self.time_order = time_order
        if self.time_order > 2:
            print("Time order only implemented up to 2. Using second order instead.")

        self.add_convection = add_convection
        self.derivative_jumps = derivative_jumps
        if derivative_jumps and order > 2:
            print("Warning: Derivative jump ghost penalty only implemented up to order 2. To use higher order ghost penalty set `derivative_jump=False`.")

        self.add_number_space = add_number_space
        self.use_supg = use_supg
        if lset is None:
            self.lset = DummyLevelSet(mesh)
        else:
            self.SetLevelSet(lset)

        self.wall_params = wall_params
        default = CF((0, 0)) if self.mesh.dim == 2 else CF((0, 0, 0))
        if f is None:
            self.f = default
        else:
            self.f = f
        self.g = g
        if surface_tension is None:
            self.surface_tension = default
        else:
            self.surface_tension = surface_tension
        self.gfup = None
        self.gfu = None
        self.gfp = None
        self.gfn = None
        self.stokes_term = None
        self.stokes_op = None
        self.conv = None
        self.conv_op = None
        self.mass = None
        self.mass_op = None
        self.m_star = None
        self.inv = None
        self.lf = None
        self.fes = None
        self.dirichlet = None
        self.neumann = None
        self.dbnd = None
        self.dt = dt
        self.nu = self.fluid_params["viscosity"]
        self.rho = self.fluid_params["density"]
        self.time = time
        self.multistepper = MultiStepper()
        self.multistepper.SetObject(self)

        self.ancient = None    # older state for bdf2
        self.boundary_registry = BoundaryRegistry()


    def Initialize(self, initial_velocity:CoefficientFunction=CF((0, 0)),
                   initial_pressure:CoefficientFunction=CF(0)):
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
        initial_velocity: CoefficientFunction
            The initial velocity field
        initial_pressure: CoefficientFunction
            The initial pressure field
        """
        self.InitializeSpaces()
        self.InitializeGridFunctions()
        self.ApplyBoundaryConditions()
        self.UpdateActiveDofs()
        self.lset.lsetadap.ProjectOnUpdate([self.current.components[0], self.current.components[1],
                                            self.intermediate.components[0], self.intermediate.components[1],
                                            self.past.components[0], self.past.components[1],
                                            self.ancient.components[0], self.ancient.components[1]],
                                           update_domain=self.els_outer)

        self.InitializeForms()
        self.SetInitialValues(initial_velocity, initial_pressure)

    def SetOuterBoundaryCondition(self, condition:BoundaryCondition):
        self.boundary_registry.AddBoundaryCondition(condition)

    def SetInnerBoundaryCondition(self, condition:typing.Union[NitscheVelocityBC, CoefficientFunction]):
        if isinstance(condition, NitscheVelocityBC):
            self.boundary_registry.AddBoundaryCondition(condition=condition)

        if isinstance(condition, CoefficientFunction):
            self.boundary_registry.AddBoundaryCondition(condition=NitscheVelocityBC(region="interface", values=condition))


    def SetInitialValues(self, initial_velocity:CoefficientFunction, initial_pressure:CoefficientFunction=CF(0),
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
        default = CF((0,0)) if self.mesh.dim == 2 else CF((0,0,0))
        cf = self.mesh.BoundaryCF(self.boundary_registry.strong_dirichlet_dict, default=default)
        self.gfu.Set(cf, definedon=self.mesh.Boundaries(self.boundary_registry.dbnd))


    def InitializeSpaces(self):
        """
        Initializes the Finite element spaces.
        """
        raise NotImplementedError("InitializeSpaces not implemented in base class.")

    def InitializeGridFunctions(self):
        """
        Initializes the grid functions.
        """
        raise NotImplementedError("InitializeGridFunctions not implemented in base class.")


    def UpdateActiveDofs(self):
        """
        Updates the active degrees of freedom after a levelset update.
        """
        raise NotImplementedError("UpdateActiveDofs not implemented in base class.")


    def InitializeForms(self):
        """
        Initializes the bilinear and linear forms.
        """
        raise NotImplementedError("InitializeForms not implemented in base class.")


    def SetLevelSet(self, lset:LevelSetGeometry):
        """
        Sets the levelset that describes the unfitted domain.
        """
        self.lset = lset
        if self.UpdateActiveDofs not in lset.callbacks:
            self.lset.callbacks.append(self.UpdateActiveDofs)
        if self.InitializeForms not in lset.callbacks:
            self.lset.callbacks.append(self.InitializeForms)


    def SolveStokes(self):
        """
        Solves the Stokes problem.

        Returns:
        ----------
        gfup: GridFunction
            The solution of the stokes problem.
        """
        raise NotImplementedError("SolveStokes not implemented in base class.")


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
        return Integrate((self.current.components[0] - self.intermediate.components[0])**2 * self.lset.dx_neg,
                         self.mesh)**(1/2)


    def Step(self):
        raise NotImplementedError("Step not implemented in base class.")


    def ValidateStep(self):
        self.ancient.vec.data = self.past.vec
        super().ValidateStep()
