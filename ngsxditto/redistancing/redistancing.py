from ngsolve import GridFunction
from abc import ABC, abstractmethod


class BaseRedistancing(ABC):
    def __init__(self, bandwidth:float=None):
        self.bandwidth = bandwidth

    @abstractmethod
    def Redistance(self, phi: GridFunction):
        """
        Applies redistancing to the given function phi.
        """
        raise NotImplementedError("Redistance not implemented for base class")


