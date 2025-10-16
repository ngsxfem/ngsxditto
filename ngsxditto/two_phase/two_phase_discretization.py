from ngsxditto import FluidParameters
from ngsxditto.stepper import *
from ngsolve import *
import typing

class TwoPhaseFlow(Stepper):
    def __init__(self, mesh: Mesh, fluid1_params: FluidParameters, fluid2_params: FluidParameters, order: int = 4,
                 lset = None, if_dirichlet:CoefficientFunction=None, wall_params: WallParameters = None,
                 f1:CoefficientFunction=None, f2: CoefficientFunction=None, g1: CoefficientFunction=CF(0),
                 g2: CoefficientFunction=CF(0),
                 surface_tension:CoefficientFunction=None, dt=None, time: typing.Optional[Parameter] = None):
        super().__init__()
        self.mesh = mesh
        self.fluid1_params = fluid1_params
        self.fluid2_params = fluid2_params
        self.order = order
        self.lset = lset
        self.if_dirichlet = if_dirichlet
        self.wall_params = wall_params
        default = CF((0, 0)) if self.mesh.dim == 2 else CF((0, 0, 0))
        if f1 is None:
            self.f1 = default
        else:
            self.f1 = f1
        if f2 is None:
            self.f2 = default
        else:
            self.f2 = f2
        self.g1 = g1
        self.g2 = g2
        if surface_tension is None:
            self.surface_tension = default
        else:
            self.surface_tension = surface_tension
        self.dt = dt
        self.time = time
