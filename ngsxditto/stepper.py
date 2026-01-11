from abc import ABC, abstractmethod
from typing import Optional, Any


class Stepper(ABC):
    """
    Abstract base class for steppers in solver loops.

    A stepper object can execute a `step`, especially in a solver loop 
    (nonlinear solvers, time loops). The `step` method is triggered 
    within such a solver loop. Additionally, it can `revert` or `validate`
    the step based on some criterion defined in the solver object.
    """

    def __init__(self):
        """
        Initialize the stepper object.
        """
        pass

    # --- Lifecycle hooks --------------------------------------------------------
    def BeforeLoop(self):
        """
        This function will be called before the solver's (outer) loop.
        Typically used to initialize objects, handle memory, etc.
        Base class implementation: do nothing
        """
        pass

    def AfterLoop(self):
        """
        This function will be called after the solver's (outer) loop.
        Typically used to write output, postprocess results, handle memory, etc.
        Base class implementation: do nothing
        """
        pass


    # --- Abstract methods that subclasses MUST implement ---------------------
    @abstractmethod
    def ValidateStep(self):
        """
        Is called at the end of each outer loop step.
        """
        pass

    @abstractmethod
    def RevertStep(self):
        """
        Is called at the end of each inner loop step if the inner loop continues.
        """
        pass


    @abstractmethod
    def Step(self):
        """
        Advances the stepper object by one (inner loop) step. What the step function
        does is defined by the subclasses.
        """
        pass



class StatefulStepper(Stepper):
    """
    Additionally to the step functions theStatefulStepper provides
    state handling via properties:
     * a `past` state,
     * an `intermediate` state and
     * a `current` state.

    Subclasses should override these if they need state management.

    The type of the states is not defined in this abstract class
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
        Initialize the stepper object by creating dummy past and intermediate states
        """
        super().__init__()
        self._past = None
        self._intermediate = None
        self._current = None

    def ValidateStep(self):
        """
        Is called at the end of each outer loop step.
        The 'current' state is validated and copied to 'past' and 'intermediate'
        states.
        """
        pass


    def RevertStep(self):
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


class StatelessStepper(Stepper):
    """
    A Stepper class where no states are needed. Subclasses must still define a step
    function. Per default do nothing in the revert and validate step stage.
    """
    def __init__(self):
        super().__init__()

    def ValidateStep(self):
        pass

    def RevertStep(self):
        pass


class FunctionCallStepper(StatelessStepper):
    """
    A stepper without states that executes provided functions at the
    relevant points of the solver loop.
    """

    def __init__(self, step_function, validate_only=False, before_loop_function=None, after_loop_function=None):
        super().__init__()
        self.step_function = step_function
        self.validate_only = validate_only
        self.BeforeLoop = before_loop_function or (lambda: None)
        self.AfterLoop = after_loop_function or (lambda: None)


    def Step(self):
        if not self.validate_only:
            self.step_function()

    def ValidateStep(self):
        if self.validate_only:
            self.step_function()


from ngsolve import GridFunction
class GFStepper(StatefulStepper):
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
    def ValidateStep(self):
        """
        Copy current -> past and intermediate (only vector entries)
        """
        if self._current is None:
            raise ValueError("current state not set")

        if self._past is None or self._intermediate is None:
            raise ValueError("past or intermediate state not initialized")

        self._past.vec.data = self._current.vec
        self._intermediate.vec.data = self._current.vec

    def RevertStep(self):
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

    def Step(self):
        """
        dummy implementation
        """
        pass

