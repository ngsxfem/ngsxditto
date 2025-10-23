from ngsxditto.fluid import *
from ngsxditto.stepper import *
from ngsolve import *
import typing

class TwoPhaseDiscretization(StatefulStepper):
    """
    Base class for two-phase fluid discretizations.
    """
    def __init__(self, mesh: Mesh, fluid1_params: FluidParameters, fluid2_params: FluidParameters, order: int = 4,
                 lset = None, if_dirichlet:CoefficientFunction=None, wall_params: WallParameters = None,
                 f1:CoefficientFunction=None, f2: CoefficientFunction=None, g1: CoefficientFunction=CF(0),
                 g2: CoefficientFunction=CF(0),
                 surface_tension:CoefficientFunction=None, dt=None, time: typing.Optional[Parameter] = None):
        super().__init__()
        self.mesh = mesh
        self.fluid1_params = fluid1_params
        self.fluid2_params = fluid2_params
        self.order = order
        if lset is None:
            self.lset = DummyLevelSet(mesh)
        else:
            self.lset = lset
            self.lset.AddCallback(self.UpdateActiveDofs)
            self.lset.AddCallback(self.InitializeCombinedSpace)
            self.lset.AddCallback(self.UpdateGfuDofs)
            #self.lset.AddCallback(self.InitializeForms)
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
        if surface_tension is None:
            self.surface_tension = default
        else:
            self.surface_tension = surface_tension
        self.dt = dt
        self.time = time
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
        self.dt = dt
        self.nu1 = self.fluid1_params["viscosity"]
        self.nu2 = self.fluid2_params["viscosity"]
        self.rho1 = self.fluid1_params["density"]
        self.rho2 = self.fluid2_params["density"]
        self.time = time
        self.multistepper = MultiStepper()
        self.multistepper.SetObject(self)


    def InitializeCombinedSpace(self):
        raise NotImplementedError("InitializeCombinedSpaces not implemened")


    def Initialize(self, dirichlet:dict=None, neumann:dict=None,
                   initial_velocity:CoefficientFunction=None,
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
        """
        default = CF((0, 0)) if self.mesh.dim == 2 else CF((0, 0, 0))
        if initial_velocity is None:
            initial_velocity = default
        self.SetBoundaryConditions(dirichlet=dirichlet, neumann=neumann)
        self.InitializeBaseSpaces()
        self.UpdateActiveDofs()
        self.InitializeCombinedSpace()
        self.InitializeGfu()
        self.ApplyBoundaryConditions()
        self.InitializeForms()
        #self.SetInitialValues(initial_velocity1, initial_velocity2, initial_pressure1, initial_pressure2)


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
                         initial_pressure1:CoefficientFunction=CF(0), initial_pressure2:CoefficientFunction=CF(0)):
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
        cf = self.mesh.BoundaryCF(self.dirichlet)
        self.gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))
        self.gfu.components[2].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))

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
        #if self.InitializeSpaces not in lset.callbacks:
        #    self.lset.AddCallback(self.InitializeSpaces())

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
        return Integrate((self.current.components[0] - self.intermediate.components[0])**2 * dx,
                         self.mesh)**(1/2)


    def Step(self):
        raise NotImplementedError("Step only implemented in subclass.")

