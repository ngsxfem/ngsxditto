from ngsxditto.callback import OnUpdateCallbacks
from ngsxditto.transport import *
from ngsxditto.redistancing import *
from xfem import *
from xfem.lsetcurv import *
from ngsolve import *
#import types


class LevelSetGeometry(OnUpdateCallbacks):
    """
    This class handles the level set geometry.
    """
    def __init__(self, transport: BaseTransport, redistancing: BaseRedistancing=None,
                 autoredistancing: AutoRedistancing=None, initial_levelset=None):
        """
        Initializes the level set object with a transport method, a redistancing method and optionally an autoredistancing scheme.
        """
        super().__init__()
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
        self.AddCallback(self.RedistanceIfNecessary)
        self.AddCallback(self.UpdateLinearApproximation)
        self.AddCallback(self.UpdateCutInfo)
        self.AddCallback(self.UpdateIntegrators)

        P1 = H1(self.mesh, order=1)
        self.lsetp1 = GridFunction(P1)

        self.lsetmeshadap = LevelSetMeshAdaptation(self.mesh, order=self.transport.order)
        self.deformation = self.lsetmeshadap.CalcDeformation(self.field)

        self.AddCallback(self.UpdateDeformation)

        self.ci = None
        self.hasif = None
        self.hasneg = None
        self.haspos = None
        self.any = None
        self.dx_neg = None
        self.dx_pos = None
        self.dS = None
        self.n = Normalize(grad(self.lsetp1))

        if initial_levelset is not None:
            self.Initialize(initial_levelset)


    def SetRedistancing(self, redistancing: BaseRedistancing):
        self.redistancing = redistancing

    def Initialize(self, initial_lset: CoefficientFunction, initial_time: float=0.0):
        self.transport.SetInitialValues(initial_lset, initial_time)
        self.UpdateLinearApproximation()
        self.UpdateCutInfo()
        self.UpdateIntegrators()


    def UpdateLinearApproximation(self):
        InterpolateToP1(self.field, self.lsetp1)

    def UpdateDeformation(self):
        self.deformation = self.lsetmeshadap.CalcDeformation(self.field)


    def UpdateCutInfo(self):
        self.ci = CutInfo(self.mesh, self.lsetp1)
        self.hasif = self.ci.GetElementsOfType(IF)
        self.hasneg = self.ci.GetElementsOfType(HASNEG)
        self.haspos = self.ci.GetElementsOfType(HASPOS)
        self.any = self.ci.GetElementsOfType(ANY)
        self.n = Normalize(grad(self.lsetp1))


    def UpdateIntegrators(self):
        self.dx_neg = dCut(levelset=self.lsetp1, domain_type=NEG, definedonelements=self.hasneg, deformation=self.deformation)
        self.dx_pos = dCut(levelset=self.lsetp1, domain_type=POS, definedonelements=self.haspos, deformation=self.deformation)
        self.dS = dCut(levelset=self.lsetp1, domain_type=IF, definedonelements=self.hasif, deformation=self.deformation)

    def OneStep(self):
        self.transport.OneStep()
        self.steps_since_last_redistancing += 1
        self.ProcessCallbacks()

    def RunFixedSteps(self, n):
        self.multistepper.RunFixedSteps(n)

    def RunUntilTime(self, end_time):
        self.multistepper.RunUntilTime(end_time)

    def ShouldRedistance(self):
        if self.autoredistancing is not None:
            return self.autoredistancing.ShouldRedistance(self)
        else:
            return False

    def Redistance(self):
        print("The next function is redistanced")
        self.redistancing.Redistance(self.field)
        self.steps_since_last_redistancing = 0

    def RedistanceIfNecessary(self):
        if self.ShouldRedistance():
            self.Redistance()

    @property
    def field(self):
        return self.transport.field