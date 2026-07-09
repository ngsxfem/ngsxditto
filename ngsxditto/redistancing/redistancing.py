from ngsolve import GridFunction
#from abc import ABC, abstractmethod

from ngsxditto.stepper import StatelessStepper


class BaseRedistancing(StatelessStepper):
    """
    This class is responsible for the abstract implementation of redistancing functionality.
    """
    def __init__(self, bandwidth:float=None):
        """
        Initialize the redistancing algorithm by settingthe bandwidth.
        """
        super().__init__()
        self.bandwidth = bandwidth
        self.field = None
        self.deformation = None
        self.order = None


    def SetOrder(self, order:int):
        """
        Set the order and adapt the algorithm if necessary.
        """
        self.order = order

    def SetField(self, field:GridFunction):
        self.field = field

    def SetDeformation(self, deformation:GridFunction):
        self.deformation = deformation

    def Redistance(self, phi: GridFunction, deformation=None):
        """
        Applies redistancing to the given function phi.

        Parameters:
        -----------
        phi: GridFunction
            The function to be redistanced.

        """
        raise NotImplementedError("Redistance not implemented for base class")

    def Step(self):
        self.Redistance(self.field, self.deformation)
