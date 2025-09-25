from ngsolve import *


class Stepper:
    def __init__(self):
        self.past = None
        self.intermediate = None


    def BeforeLoop(self):
        pass

    def AfterLoop(self):
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
        return None