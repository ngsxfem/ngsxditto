"""
This file handles fluid discretizations.
"""
from ngsolve import BilinearForm, LinearForm, Mesh
from .params import FluidParameters, WallParameters


class FluidDiscretization:
    """
    Base class for a discretized fluid.
    """
    DEFAULT_DT = 1e-3
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, levelset = None, wall_params: WallParameters = None, dt=None):
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


    def SetTimeStepSize(self):
        """
        Sets dt and reassembles necessary systems.
        """
        raise NotImplementedError("SetTimeStepSize not implemented")

    def OneStep(self):
        """
        Evolves the solution by one time step using a simple imex scheme
        """

        res = self.conv.Apply(self.gfu.vec) + self.a.mat * self.gfu.vec
        self.gfu.vec.data -= self.dt * self.inv * res

