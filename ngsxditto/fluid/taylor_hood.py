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
                 wall_params: WallParameters = None, if_dirichlet=None,
                 f: CoefficientFunction = None, g: CoefficientFunction=CF(0),
                 surface_tension: CoefficientFunction = None, dt=None, nitsche_stab:int=100,
                 ghost_stab:int=20, extension_radius:float=0.2):
        """
        Initializes the Taylor-Hood discretization with the given parameters and levelset.
        """
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, if_dirichlet=if_dirichlet, lset=lset,
                         wall_params=wall_params, f=f, g=g, surface_tension=surface_tension, dt=dt,
                         nitsche_stab=nitsche_stab, ghost_stab=ghost_stab, extension_radius=extension_radius)
        self.V = None
        self.Q = None


    def InitializeSpaces(self):
        if self.dbnd is None:
            raise TypeError("self.dbnd is still None. Set Boundary conditions first.")
        self.V = VectorH1(self.mesh, order=self.order, dirichlet=self.dbnd)
        self.Q = H1(self.mesh, order=self.order - 1)

        self.fes = FESpace([self.V, self.Q, NumberSpace(self.mesh)], dgjumps=True)
        self.gfup = GridFunction(self.fes)
        self.gfu, self.gfp, self.gfn = self.gfup.components
        self.current = self.gfup
        self.past = GridFunction(self.fes)
        self.intermediate = GridFunction(self.fes)
