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

    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, lset=None,
                 wall_params: WallParameters = None,  if_dirichlet=None, f: CoefficientFunction = CF((0, 0)),
                 surface_tension: CoefficientFunction = CF((0, 0)), dt=None, nitsche_stab=100, ghost_stab=20, extension_radius=0.2):
        """
        Initializes the Scott-Vogelius discretization with the given parameters and levelset.
        """
        if order < 4:
            print("WARNING: Scott-Vogelius for order < 4 is not stable on all meshes.")
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, lset=lset, wall_params=wall_params,
                         if_dirichlet=if_dirichlet, f=f, surface_tension=surface_tension, dt=dt,
                         nitsche_stab=nitsche_stab, ghost_stab=ghost_stab, extension_radius=extension_radius)
        self.V = None
        self.Q = None


    def InitializeSpaces(self):
        if self.dbnd is None:
            raise TypeError("self.dbnd is still None. Set Boundary conditions first.")
        self.V = VectorH1(self.mesh, order=self.order, dirichlet=self.dbnd)
        self.Q = L2(self.mesh, order=self.order - 1)
        self.fes = self.V * self.Q
        self.gfu = GridFunction(self.fes)


