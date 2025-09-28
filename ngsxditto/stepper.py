from abc import ABC, abstractmethod
from typing import Optional, Any


class Stepper(ABC):
    """
    Abstract base class for steppers in solver loops.

    A stepper object can execute a `step`, especially in a solver loop 
    (nonlinear solvers, time loops). The `step` method is triggered 
    within such a solver loop.

    Stepper provides optional state handling via properties: 
     * a `past` state,
     * an `intermediate` state and
     * a `current` state.

    Subclasses may override these if they need state management.

    The type of the states is not defined in this base class 
    and memory management is completely up to the subclass.

    The role of these properties:
     * the "past" state is the state before an outer loop step,
       typically the past in a time step loop
     * the "intermediate" state is the state before an inner loop step.
       One purpose of the intermediate state is to track the difference
       to the current state, i.e. the update in an inner loop. This 
       allows to evaluate certain stopping criteria for inner loops
     * the "current" state is the most recent state. 
    """

    def __init__(self):
        """
        Initialize the stepper object by creating dummy past and intermediate states.
        """
        self._past = None
        self._intermediate = None
        self._current = None

    # --- Lifecycle hooks --------------------------------------------------------
    def BeforeLoop(self):
        """
        This function will be called before the solver's (outer) loop.
        Typically used to initialize objects, handle memory, etc.. 
        Base class implementation: do nothing
        """
        pass

    def AfterLoop(self):
        """
        This function will be called after the solver's (outer) loop.
        Typically used to write output, postprocess results, handle memory, etc.. 
        Base class implementation: do nothing
        """
        pass

    # --- Abstract methods that subclasses MUST implement ---------------------
    @abstractmethod
    def ValidateState(self):
        """
        Is called at the end of each outer loop step. 
        The 'current' state is validated and copied to 'past' and 'intermediate'
        states.
        """
        pass

    @abstractmethod
    def RevertState(self):
        """
        Is called at the end of each inner loop step if the inner loop continues. 
        The 'current' state is copied to the 'intermediate' state.
        The 'past' state stays unaffected.
        """
        pass


    @abstractmethod
    def ComputeDifference2Intermediate(self) -> float:
        """
        Computes the difference between 'current' state and 'intermediate' state in 
        a norm defined by the subclasses.
        """
        pass


    @abstractmethod
    def Step(self):
        """
        Advances the 'current' state by one (inner loop) step.
        Does not affect the 'intermediate' state. 
        """
        pass

    # --- Optional state properties -------------------------------------------
    @property
    def current(self) -> Optional[Any]:
        return self._current

    @current.setter
    def current(self, value: Any) -> None:
        self._current = value

    @property
    def intermediate(self) -> Optional[Any]:
        return self._intermediate

    @intermediate.setter
    def intermediate(self, value: Any) -> None:
        self._intermediate = value

    @property
    def past(self) -> Optional[Any]:
        return self._past

    @past.setter
    def past(self, value: Any) -> None:
        self._past = value


class FunctionCallStepper(Stepper):
    """
    A stepper without states. Executes provided functions at the
    relevant points of the solver loop.
    """

    def __init__(self, step_function, before_loop_function=None, after_loop_function=None):
        super().__init__()
        self.Step = step_function
        self.BeforeLoop = before_loop_function or (lambda: None)
        self.AfterLoop = after_loop_function or (lambda: None)

        self.ValidateState = lambda: None
        self.RevertState = lambda: None
        self.ComputeDifference2Intermediate = lambda: 0.0


from ngsolve import GridFunction
class GFStepper(Stepper):
    """
    A Stepper whose states (past, intermediate, current) are of type ngsolve.GridFunction.
    """

    def __init__(self):
        super().__init__()

        # States are GridFunctions (initialized later)
        self._current: Optional[GridFunction] = None
        self._intermediate: Optional[GridFunction] = None
        self._past: Optional[GridFunction] = None

    # --- Abstract methods ---------------------------------------------------
    def ValidateState(self):
        """
        Copy current -> past and intermediate (only vector entries)
        """
        if self._current is None:
            raise ValueError("current state not set")

        if self._past is None or self._intermediate is None:
            raise ValueError("past or intermediate state not initialized")

        self._past.vec.data = self._current.vec
        self._intermediate.vec.data = self._current.vec

    def RevertState(self):
        """
        Copy current -> intermediate, past stays unchanged
        """
        if self._current is None:
            raise ValueError("current state not set")
        if self._intermediate is None:
            raise ValueError("intermediate state not initialized")

        self._intermediate.vec.data = self._current.vec

    def ComputeDifference2Intermediate(self) -> float:
        """
        Computes difference between current and intermediate vectors
        Uses simple l2-Norm of vectors (should be overwritten)
        """
        if self._current is None or self._intermediate is None:
            return 0.0
        diff = self._current.vec.CreateVector()
        diff.data = self._current.vec - self._intermediate.vec
        return diff.Norm()