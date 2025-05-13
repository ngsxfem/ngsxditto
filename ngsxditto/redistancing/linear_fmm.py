from ngsolve import *
from .redistancing import *


class LinearFastMarching(BaseRedistancing):
    def __init__(self, bandwidth: float=None, globally:bool=True):
        super().__init__(bandwidth, globally)
        self.order = 1

    def Redistance(self, phi: GridFunction):
        raise NotImplementedError("Redistance not yet implemented")


