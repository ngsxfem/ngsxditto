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
        Stepper.__init__(self)
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
        self.AddCallback(self.UpdateDeformation)
        self.AddCallback(self.UpdateCutInfo)

        P1 = H1(self.mesh, order=1)
        self.lsetp1 = GridFunction(P1)

        self.current = self.field
        #### TODO: generate own fes_cont space!
        if hasattr(self.transport, 'fes_cont'):
            self.past = GridFunction(self.transport.fes_cont)
            self.intermediate = GridFunction(self.transport.fes_cont)
        else:
            self.past = GridFunction(self.transport.fes)
            self.intermediate = GridFunction(self.transport.fes)


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
        #self.n = Normalize(grad(self.field))


    def DefineIntegrators(self):
        """
        Updates the integrators of the level set.
        """
        self.dx_neg = dCut(levelset=self.lsetp1, domain_type=NEG, definedonelements=self.hasneg, deformation=self.deformation)
        self.dx_pos = dCut(levelset=self.lsetp1, domain_type=POS, definedonelements=self.haspos, deformation=self.deformation)
        self.dS = dCut(levelset=self.lsetp1, domain_type=IF, definedonelements=self.hasif, deformation=self.deformation)

    def Step(self):
        """
        Evolves the level set one step with the transport scheme. Automatically updates cut info and integrators.
        """
        self.transport.Step()

        #### TODO: projiziere in stetigen Raum mit Berücksichtungen der aktiven Elemente
        # So ähnlich wie hier:
        #self.gfu_cont.Set(self.gfu, definedonelements=self.active_elements)
        #outer_cont_dofs = ~GetDofsOfElements(self.fes_cont, self.active_elements)
        #self.gfu_cont_tmp.Set(self.transport.past, definedonelements=~self.active_elements)
        #self.gfu_cont.vec += Projector(outer_cont_dofs) * self.gfu_cont_tmp.vec 


        self.gfu_cont.Set(self.transport.field)
        self.steps_since_last_redistancing += 1
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
        self.redistancing.Redistance(self.field)
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
        return self.transport.field ### take fes_cont-field here!


    def ComputeDifference2Intermediate(self):
        intermediate_gfu =GridFunction(self.current.space)
        intermediate_gfu.vec.data = self.intermediate

        error = self.current - intermediate_gfu

        interface_error = Integrate(error * error * self.dS, mesh=self.mesh) ** (1/2)
        return interface_error

