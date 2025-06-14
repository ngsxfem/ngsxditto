class MultiStepper:
    """
    This class allows handling multiple steps at the same time and checks while automatically applying redistancing.
    """
    def __init__(self):
        self.levelset = None

    def SetLevelSet(self, levelset_geometry):
        self.levelset = levelset_geometry

    def RunFixedSteps(self, n):
        for _ in range(n):
            self.levelset.OneStep()
            if self.levelset.ShouldRedistance():
                self.levelset.Redistance()

    def RunUntilTime(self, end_time):
        if self.levelset.transport.time is not None:
            while self.levelset.transport.time < end_time:
                self.levelset.OneStep()
                if self.levelset.ShouldRedistance():
                    self.levelset.Redistance()
        else:
            raise TypeError("The transport object has no time parameter")

