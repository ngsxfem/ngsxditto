

class AutoRedistancing:
    """
    This is the base class for handling auto-redistancing algorithms.
    """
    def __init__(self):
        pass

    def ShouldRedistance(self, levelset):
        """
        Checks if the levelset should be redistanced based on the auto-redistancing scheme.
        """
        raise NotImplementedError("Should Redistance not implemented")


class PeriodicRedistancing(AutoRedistancing):
    """
    Use if redistancing should be applied after a fixed amount of steps.
    """
    def __init__(self, n):
        super().__init__()
        self.n = n


    def ShouldRedistance(self, levelset):
        return levelset.steps_since_last_redistancing % self.n == 0


class GradientRedistancing(AutoRedistancing):
    """
    Use if redistancing should be applied if the level set function is out of given gradient bounds. .
    """
    def __init__(self, gradient_tester, gradient_bounds: tuple[float, float]):
        """
        Initializes the gradient based redistancing algorithm by defining the gradient testing method
        and the gradient bounds.
        Parameters:
        ----------
        gradient_tester: BaseGradientTester
            The method to obtain the gradients
        gradient_bounds: tuple[float, float]
            The gradient bounds that determine if redistancing should be applied.
        """
        super().__init__()
        self.gradient_tester = gradient_tester
        self.gradient_bounds = gradient_bounds

    def ShouldRedistance(self, levelset):
        min_grad, max_grad = self.gradient_tester.MinMaxGradientNorm(levelset.transport.field)
        return self.gradient_bounds[0] >= min_grad and self.gradient_bounds[1] <= max_grad