from ngsolve import CoefficientFunction, Mesh, Parameter

import typing 

class BaseTransport:
    """
    This class is responsible for the abstract implementation of an interface for (level-set) transport.
    """
    def __init__(self, mesh: Mesh, wind: CoefficientFunction, inflow_values: CoefficientFunction, dt: typing.Optional[float] = None, time: typing.Optional[Parameter] = None, source: typing.Optional[CoefficientFunction] = None, order:int = None):
        """
            parameters:
                mesh: computational Mesh 
                wind: velocity field CoefficientFunction
                inflow_values: CoefficientFunction for inflow boundary data
                time: reference to a Parameter for the time (to update depending coeffiecient function during propagate)
                timestepsize: time step size (if constant) that allows to do some precomputations
        """        

        self.mesh = mesh
        self.wind = wind
        self.inflow_values = inflow_values
        self.time = time
        self.dt = dt
        self.order = order
        self.source = source
        self.callbacks = []


    def SetInitialValues(self, initial_values: CoefficientFunction, initial_time: float = 0.0):
        """
        Set the initial values of the level-set function
        """
        raise NotImplementedError("SetInitialValues not implemented")


    def SetWind(self, wind: CoefficientFunction):
        """
        Set the wind velocity field that is responsible for (level-set) transport.
        """
        raise NotImplementedError("SetWind not implemented")


    def OneStep(self):
        """
        Propagate the level-set function one step with self.dt
        """
        raise NotImplementedError("OneStep not implemented")

    # deprecated! (use OneStep instead and Wrapper-Class "MultiStepper..." instead
    # for multiple time steps at once)
    def Propagate(self, t_old: float, t_new: float):
        """
        Propagate the level-set function from t_old to t_new
        """
        raise NotImplementedError("Propagate not implemented")

    @property
    def field(self):
        """
        Returns a **continuous** level-set field. This can be the GridFunction (or a part of it)
        """
        raise NotImplementedError("field not implemented")

    def AddCallBack(self, callback):
        self.callbacks.append(callback)