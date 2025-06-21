"""
This file introduces a Scott-Vogelius discretization for a fluid.
"""
from ngsolve import *

from .params import FluidParameters, WallParameters
from .h1_conforming import H1Conforming

class ScottVogelius(H1Conforming):
    """
    This class represents Scott-Vogelius elements.
    """

    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, levelset=None,
                 wall_params: WallParameters = None, dt=None):
        """
        Initializes an H1-conforming fluid represented by the Taylor-Hood element on our mesh.
        """
        if order < 4:
            print("WARNING: Scott-Vogelius for order < 4 is not stable on all meshes.")
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, levelset=levelset, wall_params=wall_params, dt=dt)


    def InitializeSpaces(self, dbnd):
        self.dbnd = dbnd
        V = VectorH1(self.mesh, order=self.order, dirichlet=dbnd)
        Q = L2(self.mesh, order=self.order - 1)
        self.fes = V * Q

