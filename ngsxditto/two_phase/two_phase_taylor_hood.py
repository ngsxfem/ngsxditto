from ngsolve import *
from xfem import *
from ngsxditto.fluid import *
from .two_phase_h1_conforming import *


class TwoPhaseTaylorHood(TwoPhaseH1Conforming):
    def __init__(self, mesh, fluid1_params: FluidParameters, fluid2_params: FluidParameters, order=4,
                 lset:LevelSetGeometry=None,wall_params: WallParameters = None, if_dirichlet:CoefficientFunction=None,
                 f1: CoefficientFunction = None, f2: CoefficientFunction = None,  g1: CoefficientFunction = CF(0),
                 g2: CoefficientFunction = CF(0),surface_tension: CoefficientFunction = None, dt=None,
                 nitsche_stab:int=100, ghost_stab:int=20, extension_radius:float=0.2):
        """
        Initializes the Two-Phase Taylor-Hood discretization with the given parameters and levelset.
        """
        super().__init__(mesh=mesh, fluid1_params=fluid1_params, fluid2_params=fluid2_params, order=order,
                         if_dirichlet=if_dirichlet, lset=lset,wall_params=wall_params, f1=f1, f2=f2, g1=g1, g2=g2,
                         surface_tension=surface_tension, dt=dt, nitsche_stab=nitsche_stab, ghost_stab=ghost_stab,
                         extension_radius=extension_radius)


    def InitializeBaseSpaces(self):
        if self.dbnd is None:
            raise TypeError("self.dbnd is still None. Set Boundary conditions first.")
        self.V_base = VectorH1(self.mesh, order=self.order, dirichlet=self.dbnd, dgjumps=True)
        self.Q_base = H1(self.mesh, order=self.order - 1)
        self.phase_fes = self.V_base * self.Q_base


    def InitializeCombinedSpace(self):
        self.fes = FESpace([
            Compress(self.V_base, self.active_u_dofs_1),
            Compress(self.Q_base, self.active_p_dofs_1),
            Compress(self.V_base, self.active_u_dofs_2),
            Compress(self.Q_base, self.active_p_dofs_2),
        ])

    def InitializeGfu(self):
        self.gfu = GridFunction(self.fes)
        self.current = self.gfu
        self.past = GridFunction(self.fes)
        self.intermediate = GridFunction(self.fes)










