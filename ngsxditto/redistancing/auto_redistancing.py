class AutoRedistancing:
    def __init__(self):
        pass

    def ShouldRedistance(self, levelset):
        raise NotImplementedError("ShouldRedistance not implemented")


class PeriodicRedistancing(AutoRedistancing):
    def __init__(self, n):
        super().__init__()
        self.n = n


    def ShouldRedistance(self, levelset):
        return levelset.steps_since_last_redistancing % self.n == 0

class GradientRedistancing(AutoRedistancing):
    def __init__(self, gradient_bounds):
        super().__init__()
        self.gradient_bounds = gradient_bounds

    def ShouldRedistance(self, levelset):
        min_grad, max_grad = levelset.MinMaxGradientNorm()
        return self.gradient_bounds[0] >= min_grad and self.gradient_bounds[1] <= max_grad