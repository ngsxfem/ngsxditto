from ngsolve import GridFunction
from abc import ABC, abstractmethod


class BaseRedistancing(ABC):
    """
    This class is responsible for the abstract implementation of redistancing functionality.
    """
    def __init__(self, bandwidth:float=None):
        """
        Initialize the redistancing algorithm by settingthe bandwidth.
        """
        self.bandwidth = bandwidth

    def SetOrder(self, order:int):
        """
        Set the order and adapt the algorithm if necessary.
        """
        raise NotImplementedError("SetOrder not implemented")

    @abstractmethod
    def Redistance(self, phi: GridFunction):
        """
        Applies redistancing to the given function phi.

        Parameters:
        -----------
        phi: GridFunction
            The function to be redistanced.

        """
        raise NotImplementedError("Redistance not implemented for base class")


