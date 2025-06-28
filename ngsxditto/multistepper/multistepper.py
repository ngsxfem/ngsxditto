class MultiStepper:
    """
    This class allows handling multiple steps at the same time and checks while automatically applying redistancing.
    """
    def __init__(self):
        self.object = None

    def SetObject(self, object):
        self.object = object

    def RunFixedSteps(self, n):
        for _ in range(n):
            self.object.OneStep()

    def RunUntilTime(self, end_time):
        if self.object.time is not None:
            while self.object.time < end_time:
                self.object.OneStep()
        else:
            raise TypeError("The transport object has no time parameter")

