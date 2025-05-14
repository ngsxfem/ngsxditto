"""
This file introduces an H1 conforming Stokes discretization for a fluid.
"""
from ngsolve import Mesh, H1, BilinearForm, LinearForm

from .discretization import FluidDiscretization
from .params import FluidParameters, WallParameters


class H1ConformingFluid(FluidDiscretization):
    """
    This class represents Taylor-Hood elements.
    """
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, levelset = None, wall_params: WallParameters = None):
        """
        Initializes an H1-conforming fluid represented by the Taylor-Hood element on our mesh.
        """
        if order < 4:
            print("WARNING: Taylor-Hood for order < 4 is not stable on all meshes.")
        super(H1ConformingFluid, self).__init__(mesh=mesh, fluid_params=fluid_params, order=order, levelset=levelset, wall_params=wall_params)

        Vh = H1(mesh, order=order) * H1(mesh, order=order-1)
        self.bf = BilinearForm(Vh)
        self.lf = LinearForm(Vh)
    