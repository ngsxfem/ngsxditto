"""
This file introduces a Taylor-Hood discretization for a fluid.
"""
from ngsolve import *

from .discretization import FluidDiscretization
from .params import FluidParameters, WallParameters
from .h1_conforming import H1Conforming
from ngsxditto.levelset import LevelSetGeometry


class TaylorHood(H1Conforming):
    """
    This class represents Taylor-Hood elements.
    """
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, dt:float=1e-2, order: int = 4, lset:LevelSetGeometry = None,
                 wall_params: WallParameters = None, add_convection:bool = False,
                 f: CoefficientFunction = None, g: CoefficientFunction=CF(0),
                 surface_tension: CoefficientFunction = None, nitsche_stab:int=100,
                 ghost_stab:int=1, extension_radius:float=0.2, derivative_jumps=False, add_number_space:bool=False,
                 time_order:int=1, use_supg:bool=False):
        """
        Initializes the Taylor-Hood discretization with the given parameters and levelset.
        """
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, lset=lset,
                         wall_params=wall_params, add_convection=add_convection, f=f, g=g,
                         surface_tension=surface_tension, dt=dt, nitsche_stab=nitsche_stab, ghost_stab=ghost_stab,
                         extension_radius=extension_radius, derivative_jumps=derivative_jumps, add_number_space=add_number_space,
                         time_order=time_order, use_supg=use_supg)
        self.V = None
        self.Q = None


    def InitializeSpaces(self):
        if self.boundary_registry.dbnd is None:
            raise TypeError("self.dbnd is still None. Set Boundary conditions first.")
        self.V = VectorH1(self.mesh, order=self.order, dirichlet=self.boundary_registry.dbnd)
        self.Q = H1(self.mesh, order=self.order - 1)
        if self.add_number_space:
            self.fes = FESpace([self.V, self.Q, NumberSpace(self.mesh)], dgjumps=True)
            self.gfup = GridFunction(self.fes)
            self.gfu, self.gfp, self.gfn = self.gfup.components
        else:
            self.fes = FESpace([self.V, self.Q], dgjumps=True)
            self.gfup = GridFunction(self.fes)
            self.gfu, self.gfp = self.gfup.components

    def InitializeGridFunctions(self):
        self.gfup = GridFunction(self.fes)

        if self.add_number_space:
            self.gfu, self.gfp, self.gfn = self.gfup.components

        else:
            self.gfu, self.gfp = self.gfup.components

        self.current = self.gfup
        self.past = GridFunction(self.fes)
        self.intermediate = GridFunction(self.fes)
        self.ancient = GridFunction(self.fes)

