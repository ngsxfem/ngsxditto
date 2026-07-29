from ngsxditto.fluid import *
from ngsxditto.stepper import *
from ngsxditto.extrapolation import Extrapolator
from ngsolve import *
import typing

LINEARIZATIONS = ("newton", "banach")


class TwoPhaseDiscretization(GFStepper):
    """
    Base class for two-phase fluid discretizations.
    """
    def __init__(self, mesh: Mesh, fluid1_params: FluidParameters, fluid2_params: FluidParameters, dt:float, order: int,
                 lset:LevelSetGeometry, wall_params: WallParameters, add_convection:bool, time_order:int,
                 f1:CoefficientFunction, f2: CoefficientFunction, g1: CoefficientFunction, g2: CoefficientFunction,
                 surface_tension_coeff:float, surface_tension:CoefficientFunction, derivative_jumps:bool,
                 add_number_space:bool, linearization:str = "newton", extrapolated_advection:bool = False,
                 time: typing.Optional[Parameter] = None):
        """
        Creates a two-phase fluid discretization on the given mesh defined by the levelset.
        If no levelset is given, create a DummyLevelSet that covers the whole domain.

        Parameters:
        -----------
        mesh: Mesh
            The computational mesh
        fluid1_params: FluidParameters
            Parameters of the first fluid (corresponding to the negative part of the levelset.)
        fluid2_params: FluidParameters
            Parameters of the second fluid (corresponding to the negative part of the levelset.)
        order: int
            the polynomial order
        lset: LevelsetGeometry
            The levelset that characterizes the unfitted domain.
        wall_params: WallParameters
            wall parameters for contact problems
        f1: CoefficientFunction
            The force term of the first phase.
        f2: CoefficientFunction
            The force term of the second phase.
        g1: CoefficientFunction
            The divergence constraint of the first phase.
        g2: CoefficientFunction
            The divergence constraint of the second phase.
        surface_tension_coeff: float
            The surface tension coefficient between the two fluids.
        surface_tension: CoefficientFunction
            The surface tension force.
        dt: float
            Time-step size
        linearization: str
            How the convective term is linearized around its advecting
            velocity beta: "newton" (default) -- the full Newton Jacobian
            grad(u)*beta + grad(beta)*u plus the residual correction N(beta);
            or "banach" -- the classical Oseen fixed-point grad(u)*beta only.
            See TwoPhaseH1Conforming for the precise schemes.
        extrapolated_advection: bool
            What beta is, orthogonal to `linearization`: False (default) uses
            the current Picard/Newton iterate (`self.intermediate`); True uses
            a history-extrapolated predictor for u^{n+1} (order 0 at start-up,
            order 1 once two validated states exist) that is *also* refed with
            the latest iterate on every sub-iteration -- an O(dt^2)-accurate
            initial guess that then converges to the same fixed point as
            `extrapolated_advection=False`, but already correct to leading
            order at n_subiter=1 (no sub-iterations needed at all).
        time: Parameter
            The time parameter.
        """
        if linearization not in LINEARIZATIONS:
            raise ValueError(f"linearization must be one of {LINEARIZATIONS}, got {linearization!r}")

        super().__init__()
        self.mesh = mesh
        self.fluid1_params = fluid1_params
        self.fluid2_params = fluid2_params
        self.order = order
        self.time_order = time_order
        if self.time_order > 2:
            print("Time order only implemented up to 2. Using second order instead.")
            self.time_order = 2
        # BDF startup counter (see fluid/discretization.py): first validated
        # step runs backward Euler.
        self.n_validated_steps = 0
        self._assembled_beta = None

        if lset is None:
            self.lset = DummyLevelSet(mesh)
        else:
            self.SetLevelSet(lset)

        self.add_convection = add_convection
        self.linearization = linearization
        self.extrapolated_advection = extrapolated_advection
        # advecting-velocity predictor (only built/used if extrapolated_advection);
        # fed with validated states (ValidateStep) and, to refine it across
        # sub-iterations, with the current iterate too (AcceptIntermediate) --
        # see _advection_velocity in TwoPhaseH1Conforming.
        self._adv_extrapolator = Extrapolator(order=1) if extrapolated_advection else None
        self._priming = False    # True only during the SetInitialValues -> ValidateStep call

        if wall_params is None:
            self.wall_params = WallParameters()
        else:
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
        self.derivative_jumps = derivative_jumps
        self.add_number_space = add_number_space
        self.surface_tension_coeff = surface_tension_coeff
        if surface_tension is None:
            self.surface_tension = default
        else:
            self.surface_tension = surface_tension
        self.dt = dt
        self.time = time
        self.gfup = None
        self.gfu = None
        self.gfp = None
        self.gfn = None
        self.stokes_op = None
        self.lf = None
        self.conv = None
        self.m_star = None
        self.inv = None
        self.mass = None
        self.stokes = None
        self.fes = None
        self.dirichlet = None
        self.neumann = None
        self.dbnd = None
        self.dt = dt
        self.nu1 = self.fluid1_params["viscosity"]
        self.nu2 = self.fluid2_params["viscosity"]
        self.rho1 = self.fluid1_params["density"]
        self.rho2 = self.fluid2_params["density"]
        self.time = time
        self.multistepper = MultiStepper()
        self.multistepper.SetObject(self)
        self.ancient = None    # older state for bdf2

        self.boundary_registry = BoundaryRegistry()


    def Initialize(self,
                   initial_velocity1:CoefficientFunction=None,
                   initial_velocity2: CoefficientFunction = None,
                   initial_pressure1:CoefficientFunction=CF(0),
                   initial_pressure2: CoefficientFunction = CF(0),
                   ):
        r"""
        Initializes the fluid discretization, setting boundary conditions of the outer as well as
        physical domain and initializing the finite element spaces and bilinear forms.
        Convenience function that combines SetBoundaryConditions, InitializeSpaces,
        ApplyBoundaryConditions, UpdateActiveDofs and InitializeForms.

        Parameters:
        -----------
        initial_velocity1: CoefficientFunction
            The initial velocity of the fluid in \Omega^{-}.
        initial_velocity2: CoefficientFunction
            The initial velocity of the fluid in \Omega^{+}.
        initial_pressure1: CoefficientFunction
            The initial pressure of the fluid in \Omega^{-}.
        initial_pressure2: CoefficientFunction
            The initial pressure of the fluid in \Omega^{+}.

        """
        default = CF((0, 0)) if self.mesh.dim == 2 else CF((0, 0, 0))
        if initial_velocity1 is None:
            initial_velocity1 = default
        if initial_velocity2 is None:
            initial_velocity2 = default
        self.InitializeBaseSpaces()
        self.UpdateActiveDofs()
        self.InitializeCombinedSpace()
        self.InitializeGfu()
        self.ApplyBoundaryConditions()
        self.lset.lsetadap.ProjectOnUpdate([self.current.components[i].components[j] for i in range(2) for j in range(2)] +
                                           [self.intermediate.components[i].components[j] for i in range(2) for j in range(2)] +
                                           [self.past.components[i].components[j] for i in range(2) for j in range(2)] +
                                           [self.ancient.components[i].components[j] for i in range(2) for j in range(2)])
        if self.extrapolated_advection:
            # zero pre-seed so the very first InitializeForms() (called next,
            # before SetInitialValues sets the real initial condition) has
            # something to Evaluate(); SetInitialValues re-feeds the same
            # (time=0) node with the true initial velocity right after.
            self._adv_extrapolator.Feed(0, self.current.components[0])
        self.InitializeForms()
        self.SetInitialValues(initial_velocity1, initial_velocity2, initial_pressure1, initial_pressure2)

    def SetOuterBoundaryCondition(self, condition:BoundaryCondition):
        self.boundary_registry.AddBoundaryCondition(condition)

    def SetInnerBoundaryCondition(self, condition:typing.Union[NitscheVelocityBC, CoefficientFunction]):
        if isinstance(condition, NitscheVelocityBC):
            self.boundary_registry.AddBoundaryCondition(condition=condition)

        if isinstance(condition, CoefficientFunction):
            self.boundary_registry.AddBoundaryCondition(condition=NitscheVelocityBC(region="interface", values=condition))


    def SetInitialValues(self, initial_velocity1:CoefficientFunction, initial_velocity2:CoefficientFunction,
                         initial_pressure1:CoefficientFunction=CF(0), initial_pressure2:CoefficientFunction=CF(0),
                         mean_pressure_fix=None):
        """
        Sets the initial values for velocity and pressure
        """
        raise NotImplementedError("SetInitialValues not implemented.")


    def ApplyBoundaryConditions(self):
        """
        Applies the boundary conditions after they are set with SetBoundaryConditions and after the spaces
        are defined with InitializeSpaces.
        """
        default = CF((0,0)) if self.mesh.dim == 2 else CF((0,0,0))
        cf = self.mesh.BoundaryCF(self.boundary_registry.strong_dirichlet_dict, default=default)
        self.gfu.components[0].Set(cf, definedon=self.mesh.Boundaries(self.boundary_registry.dbnd))
        self.gfu.components[1].Set(cf, definedon=self.mesh.Boundaries(self.boundary_registry.dbnd))


    def InitializeSpaces(self):
        """
        Initializes the Finite element spaces.
        """
        raise NotImplementedError("InitializeSpaces not implemented.")


    def UpdateActiveDofs(self):
        """
        Updates the active degrees of freedom after a levelset update.
        """
        raise NotImplementedError("UpdateActiveDofs not implemented.")


    def InitializeForms(self):
        """
        Initializes the bilinear and linear forms.
        """
        raise NotImplementedError("InitializeForms not implemented.")


    def SetLevelSet(self, lset:LevelSetGeometry):
        """
        Sets the levelset that describes the unfitted domain.
        """
        self.lset = lset
        if self.UpdateActiveDofs not in lset.callbacks:
            self.lset.callbacks.append(self.UpdateActiveDofs)
        if self.InitializeForms not in lset.callbacks:
            self.lset.callbacks.append(self.InitializeForms)

    def SetTimeStepSize(self, dt):
        """
        Sets the time step size and reassembles the necessary forms.
        """
        raise NotImplementedError("SetTimeStepSize not implemented.")


    def ComputeDifference2Intermediate(self):
        """
        return difference in velocity L2(Omega_tilde) norm where
        Omega_tilde is the background mesh
        """
        inner_integal = Integrate((self.current.components[0].components[0] -
                                   self.intermediate.components[0].components[0])**2 * self.lset.dx_neg,self.mesh)
        outer_integral = Integrate((self.current.components[0].components[1] -
                                   self.intermediate.components[0].components[1])**2 * self.lset.dx_pos,self.mesh)
        return (inner_integal + outer_integral)**(1/2)

    def Step(self):
        raise NotImplementedError("Step only implemented in subclasses.")

    def EffectiveTimeOrder(self):
        """BDF order actually used for the upcoming step (startup-aware),
        see FluidDiscretization.EffectiveTimeOrder."""
        return min(self.time_order, self.n_validated_steps + 1)

    def ValidateStep(self):
        self.ancient.vec.data = self.past.vec
        super().ValidateStep()
        self.n_validated_steps += 1
        if self.extrapolated_advection and not self._priming:
            self._adv_extrapolator.Feed(self.n_validated_steps, self.past.components[0])

    def AcceptIntermediate(self):
        super().AcceptIntermediate()
        if self.extrapolated_advection:
            # refine the "upcoming" (t^{n+1}) node with the latest Newton/Banach
            # iterate -- an extrapolate-then-interpolate predictor, see Predictor
            self._adv_extrapolator.Feed(self.n_validated_steps + 1, self.current.components[0])
