"""
This file introduces a Taylor-Hood discretization for a fluid.
"""
from ngsolve import *

from .discretization import FluidDiscretization
from .params import FluidParameters, WallParameters
from .h1_conforming import H1Conforming


class TaylorHood(H1Conforming):
    """
    This class represents Taylor-Hood elements.
    """
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, levelset = None, wall_params: WallParameters = None, dt=None):
        """
        Initializes an H1-conforming fluid represented by the Taylor-Hood element on our mesh.
        """
        if order < 4:
            print("WARNING: Taylor-Hood for order < 4 is not stable on all meshes.")
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, levelset=levelset, wall_params=wall_params, dt=dt)


    def InitializeSpaces(self, dbnd):
        self.dbnd = dbnd
        V = VectorH1(self.mesh, order=self.order, dirichlet=dbnd)
        Q = H1(self.mesh, order=self.order - 1)
        self.fes = V * Q








