"""
This file handles fluid discretizations.
"""
from ngsolve import *
from .params import FluidParameters, WallParameters
from ngsxditto.levelset import *
from ngsxditto.multistepper import MultiStepper
from ngsxditto.stepper import *
import typing


class FluidDiscretization(Stepper):
    """
    Base class for a discretized fluid.
    """
    DEFAULT_DT = 1e-3
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, lset = None,
                 if_dirichlet:CoefficientFunction=None, wall_params: WallParameters = None,
                 f:CoefficientFunction=CF((0, 0)), g: CoefficientFunction=CF(0),
                 surface_tension:CoefficientFunction=CF((0, 0)), dt=None, time: typing.Optional[Parameter] = None):
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
        if_dirichlet: CoefficientFunction
            Dirichlet boundary condition of the unfitted domain.
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
        self.if_dirichlet = if_dirichlet
        if lset is None:
            self.lset = DummyLevelSet(mesh)
        else:
            self.lset = lset
            self.lset.AddCallback(self.UpdateActiveDofs)
            self.lset.AddCallback(self.InitializeForms)
        self.wall_params = wall_params
        self.f = f
        self.g = g
        self.surface_tension = surface_tension
        self.gfu = None
        self.a = None
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
        self.dt = dt if dt is not None else self.DEFAULT_DT
        self.nu = self.fluid_params["viscosity"]
        self.rho = self.fluid_params["density"]
        self.time = time
        self.multistepper = MultiStepper()
        self.multistepper.SetObject(self)


    def Initialize(self, dirichlet:dict=None, neumann:dict=None, initial_velocity:CoefficientFunction=None, initial_pressure:CoefficientFunction=CF(0)):
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
        """
        self.SetBoundaryConditions(dirichlet=dirichlet, neumann=neumann)
        self.InitializeSpaces()
        self.ApplyBoundaryConditions()
        self.UpdateActiveDofs()
        self.InitializeForms()
        if initial_velocity is not None:
            self.SetInitialValues(initial_velocity, initial_pressure)


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


    def SetInitialValues(self, initial_velocity:CoefficientFunction, initial_pressure:CoefficientFunction=CF(0)):
        """
        Sets the initial values for velocity and pressure
        """
        raise NotImplementedError("SetInitialValues not implemented.")


    def ApplyBoundaryConditions(self):
        """
        Applies the boundary conditions after they are set with SetBoundaryConditions and after the spaces
        are defined with InitializeSpaces.
        """
        cf = self.mesh.BoundaryCF(self.dirichlet, default=CF((0, 0)))
        self.gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))


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
            lset.callbacks.append(self.UpdateActiveDofs)
        if self.InitializeForms not in lset.callbacks:
            lset.callbacks.append(self.InitializeForms)


    def SolveStokes(self):
        """
        Solves the Stokes problem.

        Returns:
        ----------
        gfu: GridFunction
            The solution of the stokes problem.
        """
        gfu = GridFunction(self.fes)
        cf = self.mesh.BoundaryCF(self.dirichlet, default=CF((0, 0)))
        gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))
        gfu.vec.data += self.a.mat.Inverse(freedofs=self.fes.FreeDofs()) * (self.lf.vec - self.a.mat * gfu.vec)
        return gfu


    def SetTimeStepSize(self, dt):
        """
        Sets the time step size and reassembles the necessary forms.
        """
        raise NotImplementedError("SetTimeStepSize not implemented.")

    @property
    def current(self):
        return self.gfu

    def ComputeDifference2Intermediate(self):
        intermediate_gfu =GridFunction(self.current.space)
        intermediate_gfu.vec.data = self.intermediate
        return Integrate((self.current.components[0] - intermediate_gfu.components[0])**2 * dx,
                         self.mesh)**(1/2)


    def ValidateState(self):
        self.past[:] = self.current.vec.data
        self.intermediate[:] = self.current.vec.data


    def RevertState(self):
        self.intermediate[:] = self.current.vec.data
        self.current.vec.data = self.past[:]


    def Step(self):
        raise NotImplementedError("Step only implemented in subclass.")

