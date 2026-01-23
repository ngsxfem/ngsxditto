from .basetransport import *
from ngsolve import *


class KnownSolutionTransport(BaseTransport):
    """
    This class handles transport for problem where the solution is known.
    """
    def __init__(self, mesh: Mesh, true_solution:CoefficientFunction, time:Parameter=None, dt:float=None, order:int=2):
        """
        Initializes the transport object with the given parameters.

        Parameters:
        -----------
        mesh: Mesh
            The computational mesh.
        true_solution: CoefficientFunction
            The true solution dependant on time.
        time: Parameter
            The time parameter of the true solution.
        dt: float
            The time-step size
        order: int
            The polynomial order of the space the solution is projected to.
        """
        super().__init__(mesh=mesh, wind=None, inflow_values=None, dt=dt, order=order)
        self.true_solution = true_solution
        self.fes = H1(mesh, order=order)
        self.gfu = GridFunction(self.fes)
        self.gfu.Set(self.true_solution)

        self.time = time

        self.current = self.gfu
        self.past = GridFunction(self.gfu.space)
        self.intermediate = GridFunction(self.gfu.space)

        self.ValidateStep()

    def SetInitialValues(self, initial_values: CoefficientFunction=None, initial_time: float = 0.0):
        pass

    def SetTimeStepSize(self, dt: float):
        self.dt = dt

    def Step(self):
        self.gfu.Set(self.true_solution)

    def AcceptIntermediate(self):
        super().AcceptIntermediate()

    def RevertStep(self):
        super().RevertStep()

    def ValidateStep(self):
        super().ValidateStep()
    
    @property
    def field(self):
        return self.gfu

