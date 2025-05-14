"""
This file handles fluid discretizations.
"""
from ngsolve import BilinearForm, LinearForm, Mesh

from .params import FluidParameters, WallParameters


class FluidDiscretization:
    """
    Base class for a discretized fluid.
    """
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, levelset = None, wall_params: WallParameters = None):
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
        self.bf = None
        self.lf = None
        self.Dbnd = Dbnd
        self.Dbndc = None

    def Initialize(self, initial_velocity):
        """
        To be honest, i do not really know what this function should do...
        """
        raise NotImplementedError("Initialize not implemented")

    def SetLevelSet(self, levelset):
        self.lset = levelset
        return self

    def DoOneStep(self, dt=0.01):
        """
        Evolutes the solution by one time step.
        """
        raise NotImplementedError("DoOneStep not implemented")
