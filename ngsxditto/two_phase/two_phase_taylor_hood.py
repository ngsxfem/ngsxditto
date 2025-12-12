from ngsolve import *
from xfem import *
from ngsxditto.fluid import *
from .two_phase_h1_conforming import *


class TwoPhaseTaylorHood(TwoPhaseH1Conforming):
    def __init__(self, mesh, fluid1_params: FluidParameters, fluid2_params: FluidParameters, order=4,
                 lset:LevelSetGeometry=None,wall_params: WallParameters = None, if_dirichlet:CoefficientFunction=None,
                 f1: CoefficientFunction = None, f2: CoefficientFunction = None,  g1: CoefficientFunction = CF(0),
                 g2: CoefficientFunction = CF(0), surface_tension: CoefficientFunction = None, dt=None,
                 add_convection:bool =False,
                 nitsche_stab:int=100, ghost_stab:int=20, extension_radius:float=0.2):
        """
        Initializes the Two-Phase Taylor-Hood discretization with the given parameters and levelset.
        """
        super().__init__(mesh=mesh, fluid1_params=fluid1_params, fluid2_params=fluid2_params, order=order,
                         if_dirichlet=if_dirichlet, lset=lset,wall_params=wall_params, f1=f1, f2=f2, g1=g1, g2=g2,
                         surface_tension=surface_tension, dt=dt, add_convection=add_convection,
                         nitsche_stab=nitsche_stab, ghost_stab=ghost_stab, extension_radius=extension_radius)

        self.V_base = None
        self.Q_base = None
        self.V_neg = None
        self.V_pos = None
        self.Q_neg = None
        self.Q_pos = None

    def InitializeBaseSpaces(self):
        """
        Initialize the base velocity and pressure space.
        """
        if self.dbnd is None:
            raise TypeError("self.dbnd is still None. Set Boundary conditions first.")
        self.V_base = VectorH1(self.mesh, order=self.order, dirichlet=self.dbnd)
        self.Q_base = H1(self.mesh, order=self.order - 1)


    def InitializeCombinedSpace(self):
        """
        Initialize the combined two-phase space depending on the dofs that correspond to each phase.
        """
        self.V_neg = Compress(self.V_base, GetDofsOfElements(self.V_base, self.els_outer))
        self.V_pos = Compress(self.V_base, GetDofsOfElements(self.V_base, ~self.els_inner))
        self.Q_neg = Compress(self.Q_base, GetDofsOfElements(self.Q_base, self.els_outer))
        self.Q_pos = Compress(self.Q_base, GetDofsOfElements(self.Q_base, ~self.els_inner))


        self.fes = FESpace([
            self.V_neg * self.V_pos,
            self.Q_neg * self.Q_pos,
            NumberSpace(self.mesh)
        ],
            dgjumps=True)

    def InitializeGfu(self):
        """
        Initializes the gfu and the GridFunctions for the stepper.
        """
        self.gfup = GridFunction(self.fes)
        self.gfu, self.gfp, self.gfn = self.gfup.components
        self.current = self.gfup
        self.past = GridFunction(self.fes)
        self.intermediate = GridFunction(self.fes)
