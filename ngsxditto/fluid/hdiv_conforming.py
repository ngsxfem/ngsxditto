from ngsolve import *

from .params import FluidParameters, WallParameters
from .discretization import FluidDiscretization


class HDivConforming(FluidDiscretization):
    def __init__(self, mesh, fluid_params: FluidParameters, order=4, lset=None, wall_params: WallParameters=None,
                 dt=None,  lamb:int=None):
        if lset is not None:
            raise NotImplementedError("HDivConforming does not support unfitted functionality. Use H1-conforming elements instead.")

        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, lset=lset, wall_params=wall_params, dt=dt)
        if lamb is None:
            self.lamb = 40*order*(order + 1)
        else:
            self.lamb = lamb

    def UpdateActiveDofs(self):
        pass

    def SetTimeStepSize(self, dt):
        self.dt = dt
        self.m_star = BilinearForm(self.fes)
        self.m_star += self.rho * self.mass
        self.m_star += self.dt * self.stokes
        self.m_star.Assemble()
        self.inv = self.m_star.mat.Inverse(self.fes.FreeDofs())


    def OneStep(self):
        res = self.conv.Apply(self.gfu.vec) + self.a.mat * self.gfu.vec
        self.gfu.vec.data -= self.dt * self.inv * res
        if self.time is not None:
            self.time += self.dt
