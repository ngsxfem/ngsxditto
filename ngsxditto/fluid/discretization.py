"""
This file handles fluid discretizations.
"""
from ngsolve import *
from .params import FluidParameters, WallParameters
from .. import MultiStepper
import typing


class FluidDiscretization:
    """
    Base class for a discretized fluid.
    """
    DEFAULT_DT = 1e-3
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, levelset = None,
                 wall_params: WallParameters = None, dt=None, time: typing.Optional[Parameter] = None):
        """
        Creates a fluid discretization on the given mesh under consideration of the levelset.
        If None is given, we simply compute the Stokes problem.

            parameters:
                mesh: mesh
                fluid_params: parameter of fluid, like viscosity
                order: polynomial order
                levelset: Levelset describing some geometry
                wall_params: wall parameters for contact problems
        """
        self.mesh = mesh
        self.fluid_params = fluid_params
        self.order = order
        self.lset = levelset
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
        self.time = time
        self.multistepper = MultiStepper()
        self.multistepper.SetObject(self)


    def SetBoundaryConditions(self, dirichlet:dict=None, neumann:dict=None):
        """
        Set the non-zero dirichlet and neumann boundary conditions for your problem.

            parameters:
                dirichlet: CoefficientFunction or similar, describing the values on the Dirichlet boundary.
                neumann: str indicating the parts of Neumann boundary
        """
        if dirichlet is None:
            dirichlet = {}

        if neumann is None:
            neumann = {}

        self.dirichlet = dirichlet
        self.neumann = neumann


    def InitializeSpaces(self, dbnd):
        raise NotImplementedError("InitializeSpaces not implemented.")


    def InitializeForms(self):
        raise NotImplementedError("InitializeForms not implemented.")


    def SetLevelSet(self, levelset):
        self.lset = levelset


    def SolveStokes(self):
        gfu = GridFunction(self.fes)
        cf = self.mesh.BoundaryCF(self.dirichlet, default=CF((0, 0)))
        gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.dbnd))
        gfu.vec.data += self.a.mat.Inverse(freedofs=self.fes.FreeDofs()) * (self.lf.vec - self.a.mat * gfu.vec)
        return gfu


    def SetTimeStepSize(self, dt):
        self.dt = dt
        self.m_star = BilinearForm(self.fes)
        self.m_star += self.mass
        self.m_star += self.dt * self.stokes
        self.m_star.Assemble()
        self.inv = self.m_star.mat.Inverse(self.fes.FreeDofs(), "sparsecholesky")

    def OneStep(self):
        """
        Evolves the solution by one time step using a simple imex scheme
        """

        res = self.conv.Apply(self.gfu.vec) + self.a.mat * self.gfu.vec
        self.gfu.vec.data -= self.dt * self.inv * res
        if self.time is not None:
            self.time += self.dt


