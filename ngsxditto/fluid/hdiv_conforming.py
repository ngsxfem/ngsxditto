from ngsolve import *

from .params import FluidParameters, WallParameters
from .discretization import FluidDiscretization


class HDivConforming(FluidDiscretization):
    def __init__(self, mesh, fluid_params: FluidParameters, order=4, levelset=None, wall_params: WallParameters=None,
                 dt=None,  lamb:int=None):
        if levelset is not None:
            raise NotImplementedError("HDivConforming does not support unfitted functionality. Use H1-conforming elements instead.")

        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, levelset=levelset, wall_params=wall_params, dt=dt)
        if lamb is None:
            self.lamb = 40*order*(order + 1)
        else:
            self.lamb = lamb