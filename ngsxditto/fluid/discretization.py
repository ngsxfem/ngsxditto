"""
This file handles fluid discretizations.
"""
from ngsolve import *
from .params import FluidParameters, WallParameters
from ngsxditto.levelset import *
from ngsxditto.multistepper import MultiStepper
import typing


class FluidDiscretization:
    """
    Base class for a discretized fluid.
    """
    DEFAULT_DT = 1e-3
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, if_dirichlet:CoefficientFunction=None,
                 lset = None, wall_params: WallParameters = None, dt=None, time: typing.Optional[Parameter] = None):
        """
        Creates a fluid discretization on the given mesh under consideration of the levelset.
        If None is given, we simply compute the Stokes problem.

            parameters:
                mesh: mesh
                fluid_params: parameter of fluid, like viscosity
                order: polynomial order
                lset: Levelset describing some geometry
                wall_params: wall parameters for contact problems
        """
        self.mesh = mesh
        self.fluid_params = fluid_params
        self.order = order
        self.if_dirichlet = if_dirichlet
        if lset is None:
            self.lset = DummyLevelSet(mesh)
        else:
            self.lset = lset
        self.wall_params = wall_params
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


    def Initialize(self, dirichlet:dict=None, neumann:dict=None, rhs=None, mean_curv=None):
        self.SetBoundaryConditions(dirichlet=dirichlet, neumann=neumann)
        self.InitializeSpaces()
        self.ApplyBoundaryConditions()
        self.UpdateActiveDofs()
        self.InitializeForms(rhs=rhs, mean_curv=mean_curv)


    def SetBoundaryConditions(self, dirichlet:dict=None, neumann:dict=None):
        """
        Set the dirichlet and neumann boundary conditions for your problem.

            parameters:
                dirichlet: dict of dirichlet boundary names (key: str) and corresponding functions (value: CoefficientFunction).
                neumann: dict of neumann boundary names (key: str) and corresponding functions (value: CoefficientFunction)
        """
        if dirichlet is None:
            dirichlet = {}

        if neumann is None:
            neumann = {}

        self.dirichlet = dirichlet
        self.neumann = neumann
        self.dbnd = "|".join(dirichlet.keys())


    def SetInitialValues(self, initial_velocity, initial_pressure=CF(0)):
        raise NotImplementedError("SetInitialValues not implemented.")


    def ApplyBoundaryConditions(self):
        cf = self.mesh.BoundaryCF(self.dirichlet, default=CF((0, 0)))
        self.gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))


    def InitializeSpaces(self):
        raise NotImplementedError("InitializeSpaces not implemented.")


    def UpdateActiveDofs(self):
        raise NotImplementedError("UpdateActiveDofs not implemented.")


    def InitializeForms(self, rhs, mean_curv):
        raise NotImplementedError("InitializeForms not implemented.")


    def SetLevelSet(self, lset):
        self.lset = lset


    def SolveStokes(self):
        gfu = GridFunction(self.fes)
        cf = self.mesh.BoundaryCF(self.dirichlet, default=CF((0, 0)))
        gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))
        gfu.vec.data += self.a.mat.Inverse(freedofs=self.fes.FreeDofs()) * (self.lf.vec - self.a.mat * gfu.vec)
        return gfu


    def SetTimeStepSize(self, dt):
        raise NotImplementedError("SetTimeStepSize not implemented.")


    def OneStep(self):
        """
        Evolves the solution by one time step using a simple imex scheme
        """
        raise NotImplementedError("OneStep not implemented.")
