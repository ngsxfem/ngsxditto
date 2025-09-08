from ngsolve import CoefficientFunction, Mesh, Parameter
from ngsxditto.multistepper import MultiStepper

import typing 

class BaseTransport:
    """
    This class is responsible for the abstract implementation of an interface for (level-set) transport.
    """
    def __init__(self, mesh: Mesh, wind: CoefficientFunction, inflow_values: CoefficientFunction,
                 dt: typing.Optional[float] = None, time: typing.Optional[Parameter] = None,
                 source: typing.Optional[CoefficientFunction] = None, order:int = None) -> None:
        """
        Initializes the transport object with the given parameters.
        Parameters:
        ----------
        mesh: Mesh
            The computational Mesh
        wind: CoefficientFunction
            The velocity field that transports the levelset
        inflow_values: CoefficientFunction
            The inflow boundary data
        time: Parameter
            reference to a Parameter for the time (to update depending coeffiecient function during propagate)
        dt: float
            The time step size for the transport.
        """        

        self.mesh = mesh
        self.wind = wind
        self.inflow_values = inflow_values
        self.time = time
        self.dt = dt
        self.order = order
        self.fes = None
        self.source = source
        self.multistepper = MultiStepper()
        self.multistepper.SetObject(self)
        self.callbacks = []


    def SetInitialValues(self, initial_values: CoefficientFunction, initial_time: float = 0.0):
        """
        Set the initial values of the level-set function.

        Parameters:
        -----------
        initial_values: CoefficientFunction
            The initial levelset function
        initial_time: float
            The initial time (default 0.0)
        """
        raise NotImplementedError("SetInitialValues not implemented")


    def SetWind(self, wind: CoefficientFunction):
        """
        Set the wind velocity field that is responsible for (level-set) transport and adapt (bi-)linearform
        if necessary.
        """
        raise NotImplementedError("SetWind not implemented")


    def SetTimeStepSize(self, dt: float):
        """
        Sets the time step size and adapt (bi-)linearform if necessary.
        """
        raise NotImplementedError("SetTimeStepSize not implemented")


    def OneStep(self):
        """
        Propagate the level-set function one step with self.dt
        """
        raise NotImplementedError("OneStep not implemented")


    @property
    def field(self):
        """
        Returns a **continuous** level-set field. This can be the GridFunction (or a part of it)
        """
        raise NotImplementedError("field not implemented")

    def AddCallBack(self, callback):
        self.callbacks.append(callback)