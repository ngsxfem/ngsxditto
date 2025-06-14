from ngsxditto.transport import *
from ngsxditto.redistancing import *
import types


class LevelSetGeometry:
    """
    This class handles the level set geometry.
    """
    def __init__(self, transport: BaseTransport, redistancing: BaseRedistancing, autoredistancing: AutoRedistancing=None):
        """
        Initializes the level set object with a transport method, a redistancing method and optionally an autoredistancing scheme.
        """
        self.transport = transport
        self.transport.SetLevelset(self)
        self.redistancing = redistancing
        self.redistancing.SetOrder(transport.order)
        self.mesh = self.transport.mesh
        self.autoredistancing = autoredistancing
        self.steps_since_last_redistancing = 0

        original_func = self.transport.OneStep
        # wrapper to track calls (useful for periodic auto redistancing)
        def wrapped_OneStep(transport):
            self.steps_since_last_redistancing += 1
            original_func()
            if self.ShouldRedistance():
                self.Redistance()

        self.transport.OneStep = types.MethodType(wrapped_OneStep, self.transport)

    def Initialize(self, initial_lset: CoefficientFunction, initial_time: float=0.0):
        self.transport.SetInitialValues(initial_lset, initial_time)

    def OneStep(self):
        self.transport.OneStep()

    def RunFixedSteps(self, n):
        self.transport.multistepper.RunFixedSteps(n)

    def RunUntilTime(self, end_time):
        self.transport.multistepper.RunUntilTime(end_time)

    def ShouldRedistance(self):
        if self.autoredistancing is not None:
            return self.autoredistancing.ShouldRedistance(self)
        else:
            return False

    def Redistance(self):
        print("The next function is redistanced")
        self.redistancing.Redistance(self.transport.field)
        self.steps_since_last_redistancing = 0


    def MinMaxGradientNorm(self, bandwidth=None):
        phi = self.transport.field
        norm_grad = Norm(grad(phi))
        V = phi.space
        gfu = GridFunction(V)
        gfu.Set(norm_grad)

        max_grad = -1e100
        min_grad = 1e100

        for v in self.mesh.vertices:
            point = self.mesh[v].point
            val = gfu(self.mesh(*point))
            max_grad = max(max_grad, val)
            min_grad = min(min_grad, val)

        return min_grad, max_grad
