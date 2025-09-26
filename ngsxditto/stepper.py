from ngsolve import *


class Stepper:
    """
    The stepper class keeps track of the past, intermediate and current state.
    """
    def __init__(self):
        """
        Initialize the stepper object by creating dummy past and intermediate states.
        """
        self.past = None
        self.intermediate = None


    def BeforeLoop(self):
        """
        This function will be called before the solver loop for each object.
        """
        pass

    def AfterLoop(self):
        """
        This function will be called after the solver loop for each object.
        """
        pass


    def ValidateState(self):
        """
        Sets 'past' and 'intermediate' to the current State
        """
        raise NotImplementedError("ValidateState only implemented in subclass")

    def RevertState(self):
        """
        Saves the current State in 'intermediate'. Resets the current State to 'past'.
        """
        raise NotImplementedError("ResetState only implemented in subclass")


    def ComputeDifference2Intermediate(self):
        """
        Computes the difference between current state and intermediate state in a norm defined by the subclasses.
        """
        raise NotImplementedError("ComputeDifference2Intermediate only implemented in subclass.")


    def Step(self):
        """
        Advances the current State by one step.
        """
        raise NotImplementedError("Step only implemented in subclass.")


    @property
    def current(self):
        """
        The current state of the solver defined by the respective subclass.
        """
        return None