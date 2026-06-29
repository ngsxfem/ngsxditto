import logging
from ngsxditto.callback import OnUpdateCallbacks
from ngsxditto.transport import *
from ngsxditto.redistancing import *
from xfem import *
from xfem.lsetcurv import *
from ngsolve import *
from ngsxditto.stepper import *

logger = logging.getLogger(__name__)

#import types


class LevelSetGeometry(OnUpdateCallbacks, GFStepper):
    """
    This class handles the level set geometry.
    """
    def __init__(self, transport: BaseTransport, redistancing: BaseRedistancing=None,
                 autoredistancing: AutoRedistancing=None, initial_levelset:CoefficientFunction=None,
                 boundary_tangential=False):
        """
        Initializes the level set object with a transport method, a redistancing method and optionally an
        autoredistancing scheme. Automatically adds callbacks that update cut info and integrators every
        levelset update.

        Parameters:
        -----------
        transport : BaseTransport
            The method for transporting the levelset.
        redistancing : BaseRedistancing
            The redistancing method.
        autoredistancing : AutoRedistancing
            The autoredistancing scheme, i.e. when redistancing should be applied.
        initial_levelset: CoefficientFunction
            The initial levelset function.
        boundary_tangential : bool / str / list / Region
            Forwarded to LevelSetMeshAdaptation. Keeps the isoparametric
            deformation tangential to the named boundaries (u.n = 0) where the
            zero level set crosses the domain boundary, so higher-order cut
            accuracy is preserved for interfaces meeting the wall (contact line).
        """
        OnUpdateCallbacks.__init__(self)
        GFStepper.__init__(self)
        self.transport = transport
        self.time = self.transport.time
        self.multistepper = MultiStepper()
        self.multistepper.SetObject(self)
        self.mesh = self.transport.mesh
        self.autoredistancing = autoredistancing
        self.steps_since_last_redistancing = 0
        self.order = transport.order
        P1 = H1(self.mesh, order=1)
        self.lsetp1 = GridFunction(P1)

        self.fes_cont = H1(self.mesh, order=self.order)

        self.lset_cont = GridFunction(self.fes_cont)
        self.lset_cont_tmp = GridFunction(self.fes_cont)


        self.current = self.lset_cont # current points to lset_cont
        self.past = GridFunction(self.fes_cont)
        self.intermediate = GridFunction(self.fes_cont)

        self.lsetadap = LevelSetMeshAdaptation(self.mesh, order=self.transport.order,
                                               boundary_tangential=boundary_tangential)
        self.deformation = self.lsetadap.deform
        if redistancing is not None:
            self.redistancing = redistancing
            self.redistancing.SetOrder(transport.order)
            self.redistancing.SetField(self.lsetp1)

        self.cutinfo = CutInfo(self.mesh)
        self.hasif = self.cutinfo.GetElementsOfType(IF)
        self.hasneg = self.cutinfo.GetElementsOfType(HASNEG)
        self.haspos = self.cutinfo.GetElementsOfType(HASPOS)
        self.any = self.cutinfo.GetElementsOfType(ANY)

        self.dx_neg = None
        self.dx_pos = None
        self.dS = None
        self.n = Normalize(grad(self.lsetp1))

        if initial_levelset is not None:
            self.Initialize(initial_levelset)

    def ValidateStep(self):
        self.steps_since_last_redistancing += 1
        self.RedistanceIfNecessary()
        self.transport.ValidateStep()
        super().ValidateStep()

        if hasattr(self.transport, 'past_cont'):
            self.transport.past_cont.vec.data = self.field.vec

    def AcceptIntermediate(self):
        self.transport.AcceptIntermediate()
        self.intermediate.vec.data = self.current.vec
        self.current.vec.data = self.past.vec


    def RevertStep(self):
        self.transport.RevertStep()
        super().AcceptIntermediate()


    @classmethod
    def from_cf(cls, cf : CoefficientFunction, mesh : Mesh, order : int = 1 ):
        """
            Initializes a LevelSetGeometry from a CoefficientFunction using a NoTransport
            object for the transport
        """
        return cls(transport=NoTransport(mesh, order=order), initial_levelset=cf)


    def SetRedistancing(self, redistancing: BaseRedistancing):
        """
        Sets the redistancing method.
        """
        self.redistancing = redistancing


    def Initialize(self, initial_lset: CoefficientFunction, initial_time: float=0.0):
        """
        Initializes the level set object.
        Convenience function that sets linear approximation, the deformation, the cut info and the integrators.

        Parameters:
        -----------
        initial_lset : CoefficientFunction
            The initial levelset function.
        initial_time : float
            The initial time. (Default: 0.0)
        """
        self.transport.SetInitialValues(initial_lset, initial_time)
        self.ProjectToContinuous(whole_mesh=True)

        self.UpdateLinearApproximation()
        self.UpdateDeformation()
        self.UpdateCutInfo()
        self.DefineIntegrators()
        self.ValidateStep()
        self.steps_since_last_redistancing = 0


    def UpdateLinearApproximation(self):
        """
        Updates the linear approximation of the level set.
        """
        InterpolateToP1(self.field, self.lsetp1)

    def UpdateDeformation(self):
        """
        Updates the deformation of the level set.
        """
        self.lsetadap.CalcDeformation(self.field)


    def UpdateCutInfo(self):
        """
        Updates the cut info of the level set.
        """
        self.cutinfo.Update(self.lsetp1)
        self._DropDetachedRegions()

    def _DropDetachedRegions(self):
        """
        Removes spurious far-field regions from the element markings
        (in place, so that integrators and consumers holding references to
        hasneg/hasif see the cleaned sets). Away from the interface the level
        set values are not controlled (they are advected by an extension wind
        and possibly never redistanced), so perturbations can spuriously
        cross zero and create small islands detached from the tracked
        geometry. Only the facet-connected components of hasneg that overlap
        the geometry's position at the previous update are kept; for
        CFL-bounded transport the geometry moves less than one element layer
        per step, so this overlap is always non-empty for the true geometry.
        """
        import os
        if os.environ.get("NGSXDITTO_DISABLE_MARKER_FILTERS"):
            return

        def flood(seeds):
            reached = BitArray(seeds)
            while True:
                front = GetFacetsWithNeighborTypes(self.mesh, a=reached, b=self.hasneg)
                grown = GetElementsWithNeighborFacets(self.mesh, front)
                new = grown & self.hasneg & ~reached
                if new.NumSet() == 0:
                    break
                reached |= new
            return reached

        # components made up of cut elements only (no uncut interior element)
        # are below resolution: they cannot be element-aggregated and their
        # geometry is not meaningful on this mesh
        uncut_neg = self.hasneg & ~self.hasif
        if uncut_neg.NumSet() > 0:
            reached = flood(uncut_neg)
            n_dropped = (self.hasneg & ~reached).NumSet()
            if n_dropped > 0:
                logger.debug("dropped %d cut element(s) of sub-resolution "
                             "level set regions", n_dropped)
                self.hasneg &= reached
                self.hasif &= reached

        if getattr(self, "_prev_hasneg", None) is not None:
            seeds = self._prev_hasneg & self.hasneg
            if seeds.NumSet() > 0:
                reached = flood(seeds)
                n_dropped = (self.hasneg & ~reached).NumSet()
                if n_dropped > 0:
                    logger.debug("dropped %d element(s) of detached spurious "
                                 "level set regions", n_dropped)
                    self.hasneg &= reached
                    self.hasif &= reached
        self._prev_hasneg = BitArray(self.hasneg)


    def DefineIntegrators(self):
        """
        Updates the integrators of the level set.
        """
        self.dx_neg = dCut(levelset=self.lsetp1, domain_type=NEG, definedonelements=self.hasneg, deformation=self.deformation)#, order=self.transport.order)
        self.dx_pos = dCut(levelset=self.lsetp1, domain_type=POS, definedonelements=self.haspos, deformation=self.deformation)#, order=self.transport.order)
        self.dS = dCut(levelset=self.lsetp1, domain_type=IF, definedonelements=self.hasif, deformation=self.deformation)#, order=self.transport.order)

    def ProjectToContinuous(self, whole_mesh=False):
        """
        Projects the transport field to the continuous level set.
        """
        if whole_mesh or self.transport.active_elements is None:
            self.lset_cont.Set(self.transport.field)
        else:
            # first take values on active elements
            self.lset_cont.Set(self.transport.field, definedonelements=self.transport.active_elements)
            # take values from old lset on the remainder **without** changing the active elements.
            outer_cont_dofs = ~GetDofsOfElements(self.fes_cont, self.transport.active_elements)
            self.lset_cont_tmp.Set(self.past, definedonelements=~self.transport.active_elements)
            self.lset_cont.vec.data += Projector(outer_cont_dofs,range=True) * self.lset_cont_tmp.vec

    @timed_method
    def Step(self):
        """
        Evolves the level set one step with the transport scheme. Automatically updates cut info and integrators.
        """

        self.transport.Step() # step on auxiliary field (e.g. DG)
        self.ProjectToContinuous()
        self.UpdateLinearApproximation()
        self.UpdateCutInfo()
        self.UpdateDeformation()

        self.ProcessCallbacks()

    def RunFixedSteps(self, n):
        """
        Runs a fixed number of steps.
        """
        self.multistepper.RunFixedSteps(n)

    def RunUntilTime(self, end_time):
        """
        Runs until the time object reaches given value.
        """
        self.multistepper.RunUntilTime(end_time)

    def ShouldRedistance(self):
        """
        Checks if the redistancing algorithm should be applied based on the autoredistancing scheme.
        """
        if self.autoredistancing is not None:
            return self.autoredistancing.ShouldRedistance(self)
        else:
            return False

    @timed_method
    def Redistance(self):
        """
        Applies the redistancing algorithm.
        """
        print("The next function is redistanced")


        old_lsetp1 = GridFunction(H1(self.mesh, order=1))
        old_lsetp1.vec.data = self.lsetp1.vec
        self.redistancing.Step()

        #ProjectShift(self.field, self.lsetp1, self.deformation, qn=self.field.Deriv(),
        #             lower=0.0, upper=0.0, threshold=-1.0, heapsize=1000000)

        tmp_field = GridFunction(self.fes_cont)
        tmp_field.Set(shifted_eval(self.lsetp1, back=self.deformation, forth=None), definedonelements=self.hasif)
        self.field.Set(shifted_eval(self.lsetp1, back=self.deformation, forth=None))
        hasif_dofs = GetDofsOfElements(self.fes_cont, self.hasif)
        self.field.vec.data[hasif_dofs] = Projector(hasif_dofs, range=True) * tmp_field.vec
        self.UpdateLinearApproximation()
        self.UpdateDeformation()
        self.transport.field.Set(self.field)
        self.ProcessCallbacks()

        self.steps_since_last_redistancing = 0

    def RedistanceIfNecessary(self):
        """
        Apllies the redistancing algorithm if it should be applied based on the autoredistancing scheme.
        """
        if self.ShouldRedistance():
            self.Redistance()

    @property
    def surface_area(self):
        return Integrate(CF(1) * dCut(levelset=self.lsetp1, domain_type=IF, definedonelements=self.hasif,
                                      deformation=self.deformation, order=self.transport.order), self.mesh)


    @property
    def volume(self):
        return Integrate(CF(1) * dCut(levelset=self.lsetp1, domain_type=NEG, definedonelements=self.hasneg, deformation=self.deformation, order=self.transport.order), self.mesh)

    @property
    def field(self):
        return self.lset_cont


    def ComputeDifference2Intermediate(self):
        error = self.current - self.intermediate

        interface_error = Integrate(error * error * self.dS, mesh=self.mesh) ** (1/2)
        return interface_error
