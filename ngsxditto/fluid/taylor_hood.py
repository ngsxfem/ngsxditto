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
        self.free_dofs = None


    def InitializeSpaces(self):
        if self.boundary_registry.dbnd is None:
            raise TypeError("self.dbnd is still None. Set Boundary conditions first.")
        self.V = VectorH1(self.mesh, order=self.order, dirichlet=self.boundary_registry.dbnd)
        self.Q = H1(self.mesh, order=self.order - 1)
        if self.add_number_space:
            self.fes = FESpace([self.V, self.Q, NumberSpace(self.mesh)], dgjumps=True)
        else:
            self.fes = FESpace([self.V, self.Q], dgjumps=True)
        self.free_dofs = self.fes.FreeDofs()
        components = list(self.V.components)
        normal_dofs = BitArray(self.fes.ndof)
        normal_dofs[:] = False

        if self.mesh.dim == 2:
            self.V_x, self.V_y = components
            offset_y = self.V_x.ndof

            bnd_x = self.V_x.GetDofs(self.mesh.Boundaries("left|right"))
            bnd_y = self.V_y.GetDofs(self.mesh.Boundaries("top|bottom"))

            for i, is_on_bnd in enumerate(bnd_x):
                if is_on_bnd:
                    normal_dofs[i] = True
            for i, is_on_bnd in enumerate(bnd_y):
                if is_on_bnd:
                    normal_dofs[offset_y + i] = True
        else:
            self.V_x, self.V_y, self.V_z = components
            offset_y = self.V_x.ndof
            offset_z = self.V_x.ndof + self.V_y.ndof

            _bnd_names = self.mesh.GetBoundaries()
            x_bnds = "|".join(b for b in _bnd_names if b in ("left", "right"))
            y_bnds = "|".join(b for b in _bnd_names if b in ("top", "bottom"))
            z_bnds = "|".join(b for b in _bnd_names if b in ("front", "back"))

            if x_bnds:
                for i, v in enumerate(self.V_x.GetDofs(self.mesh.Boundaries(x_bnds))):
                    if v: normal_dofs[i] = True
            if y_bnds:
                for i, v in enumerate(self.V_y.GetDofs(self.mesh.Boundaries(y_bnds))):
                    if v: normal_dofs[offset_y + i] = True
            if z_bnds:
                for i, v in enumerate(self.V_z.GetDofs(self.mesh.Boundaries(z_bnds))):
                    if v: normal_dofs[offset_z + i] = True
        zero_normal_region = "|".join(self.boundary_registry.strong_normal_velocity_dict.keys())
        zero_normal_dofs = normal_dofs & self.V.GetDofs(self.mesh.Boundaries(zero_normal_region))
        self.free_dofs &= ~zero_normal_dofs


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

