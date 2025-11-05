from ngsolve import CoefficientFunction, Mesh, Parameter, BitArray
from ngsxditto.multistepper import MultiStepper
from ngsxditto.stepper import GFStepper

import typing 

class BaseTransport(GFStepper):
    """
    This class is responsible for the abstract implementation of an interface for (level-set) transport.
    """
    def __init__(self, mesh: Mesh, wind: CoefficientFunction, inflow_values: CoefficientFunction,
                 dt: typing.Optional[float] = None, 
                 time: typing.Optional[Parameter] = None, ## TODO: to remove
                 source: typing.Optional[CoefficientFunction] = None, 
                 active_elements: typing.Optional[BitArray] = None,
                 order:int = None) -> None:
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
        source: CoefficientFunction 
            The source term
        active_elements: BitArray|None
            submesh defined by BitArray on which the transport is defined. If None, the whole mesh is used.
        dt: float
            The time step size for the transport.
        """        
        super().__init__()

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
        self.active_elements = active_elements


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


    def Step(self):
        """
        Propagate the level-set function one step with self.dt
        """
        raise NotImplementedError("UpdateStates not implemented")


    @property
    def field(self):
        """
        Returns a **continuous** level-set field. This can be the GridFunction (or a part of it)
        """
        raise NotImplementedError("field not implemented")