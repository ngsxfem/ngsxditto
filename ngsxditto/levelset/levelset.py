from ngsxditto.callback import OnUpdateCallbacks
from ngsxditto.transport import *
from ngsxditto.redistancing import *
from xfem import *
from xfem.lsetcurv import *
from ngsolve import *
from ngsxditto.stepper import *

#import types


class LevelSetGeometry(OnUpdateCallbacks, GFStepper):
    """
    This class handles the level set geometry.
    """
    def __init__(self, transport: BaseTransport, redistancing: BaseRedistancing=None,
                 autoredistancing: AutoRedistancing=None, initial_levelset:CoefficientFunction=None):
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
        """
        OnUpdateCallbacks.__init__(self)
        GFStepper.__init__(self)
        self.transport = transport
        self.time = self.transport.time
        self.multistepper = MultiStepper()
        self.multistepper.SetObject(self)
        if redistancing is not None:
            self.redistancing = redistancing
            self.redistancing.SetOrder(transport.order)
        self.mesh = self.transport.mesh
        self.autoredistancing = autoredistancing
        self.steps_since_last_redistancing = 0

        P1 = H1(self.mesh, order=1)
        self.lsetp1 = GridFunction(P1)


        self.fes_cont = H1(self.mesh, order=self.transport.order)

        self.lset_cont = GridFunction(self.fes_cont)
        self.lset_cont_tmp = GridFunction(self.fes_cont)


        self.current = self.lset_cont # current points to lset_cont
        self.past = GridFunction(self.fes_cont)
        self.intermediate = GridFunction(self.fes_cont)

        self.lsetadap = LevelSetMeshAdaptation(self.mesh, order=self.transport.order)
        self.deformation = self.lsetadap.deform

        self.cutinfo = CutInfo(self.mesh)
        self.hasif = self.cutinfo.GetElementsOfType(IF)
        self.hasneg = self.cutinfo.GetElementsOfType(HASNEG)
        self.haspos = self.cutinfo.GetElementsOfType(HASPOS)
        self.any = self.cutinfo.GetElementsOfType(ANY)

        self.dx_neg = None
        self.dx_pos = None
        self.dS = None
        self.n = Normalize(grad(self.field))

        if initial_levelset is not None:
            self.Initialize(initial_levelset)

    def ValidateStep(self):
        self.transport.ValidateStep()
        super().ValidateStep()

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


    def DefineIntegrators(self):
        """
        Updates the integrators of the level set.
        """
        self.dx_neg = dCut(levelset=self.lsetp1, domain_type=NEG, definedonelements=self.hasneg, deformation=self.deformation)
        self.dx_pos = dCut(levelset=self.lsetp1, domain_type=POS, definedonelements=self.haspos, deformation=self.deformation)
        self.dS = dCut(levelset=self.lsetp1, domain_type=IF, definedonelements=self.hasif, deformation=self.deformation)

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

    def Step(self):
        """
        Evolves the level set one step with the transport scheme. Automatically updates cut info and integrators.
        """

        self.transport.Step() # step on auxiliary field (e.g. DG)
        self.ProjectToContinuous()
        self.steps_since_last_redistancing += 1
        self.RedistanceIfNecessary()
        self.UpdateLinearApproximation()
        self.UpdateDeformation()
        self.UpdateCutInfo()

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

    def Redistance(self):
        """
        Applies the redistancing algorithm.
        """
        print("The next function is redistanced")
        self.redistancing.Redistance(self.transport.field)
        self.ProjectToContinuous()
        self.steps_since_last_redistancing = 0

    def RedistanceIfNecessary(self):
        """
        Apllies the redistancing algorithm if it should be applied based on the autoredistancing scheme.
        """
        if self.ShouldRedistance():
            self.Redistance()

    @property
    def surface_area(self):
        return Integrate(CF(1) * self.dS, self.mesh)

    @property
    def volume(self):
        return Integrate(CF(1) * self.dx_neg, self.mesh)

    @property
    def field(self):
        return self.lset_cont


    def ComputeDifference2Intermediate(self):
        error = self.current - self.intermediate

        interface_error = Integrate(error * error * self.dS, mesh=self.mesh) ** (1/2)
        return interface_error

