from ngsolve import *
from .redistancing import *
from .linear_fmm import *
from .quadratic_fmm import *


class FastMarching(BaseRedistancing):
    """
    This class handles redistancing using the Fast Matching Method. Implemented only for orders 1 and 2.
    Approximately restores the signed distance property globally or in a bandwidth around the level set.
    """
    def __init__(self, bandwidth: float=None, order: int=None):
        """
        Initializes the redistancing algorithm by checking the order.

        Parameters:
        -----------
        bandwidth: float
            The bandwith around the levelset where redistancing should be applied.
        order: int
            The order of the redistancing algorithm.
        """
        super().__init__(bandwidth)
        self.order = order
        if self.order is not None:
            if self.order == 1:
                self.redistancing_algorithm = LinearFastMarching(bandwidth)
            elif self.order == 2:
                self.redistancing_algorithm = QuadraticFastMarching(bandwidth)
            else:
                raise NotImplementedError("FastMarching only supports order 1 and 2")
        else:
            self.redistancing_algorithm = None

    def SetOrder(self, order: int):
        if order > 2:
            raise NotImplementedError("FastMarching only supports order 1 and 2")

        self.order = order

        if self.order == 1:
            self.redistancing_algorithm = LinearFastMarching(self.bandwidth)
        if self.order == 2:
            self.redistancing_algorithm = QuadraticFastMarching(self.bandwidth)

    def Redistance(self, phi: GridFunction):
        self.redistancing_algorithm.Redistance(phi)
