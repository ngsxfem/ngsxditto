


class AutoRedistancing:
    def __init__(self):
        pass

    def ShouldRedistance(self):
        raise NotImplementedError("ShouldRedistance not implemented")


class PeriodicRedistancing(AutoRedistancing):
    def __init__(self, n):
        self.n = n

    def ShouldRedistance(self, step):
        return step % self.interval == 0