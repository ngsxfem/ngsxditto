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
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, lset = None,
                 wall_params: WallParameters = None, dt=None, sigma=100, ghost_stab=20, delta=0.2):
        """
        Initializes an H1-conforming fluid represented by the Taylor-Hood element on our mesh.
        """
        if order < 4:
            print("WARNING: Taylor-Hood for order < 4 is not stable on all meshes.")
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, lset=lset, wall_params=wall_params, dt=dt,
                         sigma=sigma, ghost_stab=ghost_stab, delta=delta)
        self.V = None
        self.Q = None


    def InitializeSpaces(self):
        if self.dbnd is None:
            raise TypeError("self.dbnd is still None. Set Boundary conditions first.")
        self.V = VectorH1(self.mesh, order=self.order, dirichlet=self.dbnd, dgjumps=True)
        self.Q = H1(self.mesh, order=self.order - 1)
        self.fes = self.V * self.Q
        self.gfu = GridFunction(self.fes)









