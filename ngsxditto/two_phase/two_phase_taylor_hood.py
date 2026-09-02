from ngsolve import *
from xfem import *
from ngsxditto.fluid import *
from .two_phase_h1_conforming import *
import typing


class TwoPhaseTaylorHood(TwoPhaseH1Conforming):
    def __init__(self, mesh: Mesh, fluid1_params: FluidParameters, fluid2_params: FluidParameters, dt:float, order:int=4,
                 lset:LevelSetGeometry=None, wall_params: WallParameters = None, time_order:int=1,
                 f1: CoefficientFunction = None, f2: CoefficientFunction = None,  g1: CoefficientFunction = CF(0),
                 g2: CoefficientFunction = CF(0), advection:typing.Union[bool, CoefficientFunction]=True,
                 surface_tension_coeff:float=1., surface_tension: CoefficientFunction = None,
                 derivative_jumps:bool=False, add_number_space:bool=False,
                 nitsche_stab:int=100, ghost_stab:int=20, extension_radius:float=0.2,
                 linearization:str = "newton", extrapolated_advection:bool = False):
        """
        Initializes the Two-Phase Taylor-Hood discretization with the given parameters and levelset.

        linearization: str
            How the convective term is linearized around its advecting
            velocity beta: "newton" (default) or "picard" -- see
            TwoPhaseH1Conforming.__init__ for the precise schemes.
        extrapolated_advection: bool
            What beta is, orthogonal to `linearization`: False (default) uses
            the current Picard/Newton iterate; True uses a history-extrapolated,
            sub-iteration-refined predictor for u^{n+1} -- see
            TwoPhaseH1Conforming.__init__.
        """
        super().__init__(mesh=mesh, fluid1_params=fluid1_params, fluid2_params=fluid2_params, order=order,
                         lset=lset,wall_params=wall_params, f1=f1, f2=f2, g1=g1, g2=g2, time_order=time_order,
                         surface_tension_coeff=surface_tension_coeff,
                         surface_tension=surface_tension, dt=dt, advection=advection,
                         nitsche_stab=nitsche_stab, ghost_stab=ghost_stab, extension_radius=extension_radius,
                         derivative_jumps=derivative_jumps, add_number_space=add_number_space,
                         linearization=linearization, extrapolated_advection=extrapolated_advection)

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
        if self.boundary_registry.dbnd is None:
            raise TypeError("self.dbnd is still None. Set Boundary conditions first.")
        self.V_base = VectorH1(self.mesh, order=self.order, dirichlet=self.boundary_registry.dbnd)
        self.Q_base = H1(self.mesh, order=self.order - 1)


    def InitializeCombinedSpace(self):
        """
        Initialize the combined two-phase space depending on the dofs that correspond to each phase.
        """
        #self.V_neg = Compress(self.V_base, GetDofsOfElements(self.V_base, self.els_outer))
        #self.V_pos = Compress(self.V_base, GetDofsOfElements(self.V_base, ~self.els_inner))
        #self.Q_neg = Compress(self.Q_base, GetDofsOfElements(self.Q_base, self.els_outer))
        #self.Q_pos = Compress(self.Q_base, GetDofsOfElements(self.Q_base, ~self.els_inner))
        self.V_neg = self.V_base
        self.V_pos = self.V_base
        self.Q_neg = self.Q_base
        self.Q_pos = self.Q_base

        if self.add_number_space:
            self.fes = FESpace([
                self.V_neg * self.V_pos,
                self.Q_neg * self.Q_pos,
                NumberSpace(self.mesh) * NumberSpace(self.mesh)
            ],
                dgjumps=True)
        else:
            self.fes = FESpace([
                self.V_neg * self.V_pos,
                self.Q_neg * self.Q_pos,
            ],
                dgjumps=True)
        self.free_dofs = self.fes.FreeDofs()
        components = list(self.V_base.components)
        normal_dofs = BitArray(self.fes.ndof)
        normal_dofs[:] = False

        offset_phase2 = self.V_base.ndof

        if self.mesh.dim == 2:
            self.V_x, self.V_y = components
            offset_y = self.V_x.ndof

            bnd_x = self.V_x.GetDofs(self.mesh.Boundaries("left|right"))
            bnd_y = self.V_y.GetDofs(self.mesh.Boundaries("top|bottom"))

            for i, is_on_bnd in enumerate(bnd_x):
                if is_on_bnd:
                    normal_dofs[i] = True
                    normal_dofs[offset_phase2 + i] = True
            for i, is_on_bnd in enumerate(bnd_y):
                if is_on_bnd:
                    normal_dofs[offset_y + i] = True
                    normal_dofs[offset_phase2 + offset_y + i] = True
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
                    if v:
                        normal_dofs[i] = True
                        normal_dofs[offset_phase2 + i] = True
            if y_bnds:
                for i, v in enumerate(self.V_y.GetDofs(self.mesh.Boundaries(y_bnds))):
                    if v:
                        normal_dofs[offset_y + i] = True
                        normal_dofs[offset_phase2 + offset_y + i] = True
            if z_bnds:
                for i, v in enumerate(self.V_z.GetDofs(self.mesh.Boundaries(z_bnds))):
                    if v:
                        normal_dofs[offset_z + i] = True
                        normal_dofs[offset_phase2 + offset_z + i] = True
        zero_normal_region = "|".join(self.boundary_registry.strong_normal_velocity_dict.keys())
        zero_normal_dofs = normal_dofs & self.fes.GetDofs(self.mesh.Boundaries(zero_normal_region))
        self.free_dofs &= ~zero_normal_dofs


    def InitializeGfu(self):
        """
        Initializes the gfu and the GridFunctions for the stepper.
        """
        self.gfup = GridFunction(self.fes)
        self.gfu, self.gfp = self.gfup.components[0], self.gfup.components[1]
        self.gfn = self.gfup.components[2] if self.add_number_space else None
        self.current = self.gfup
        self.past = GridFunction(self.fes)
        self.intermediate = GridFunction(self.fes)
        self.ancient = GridFunction(self.fes)

