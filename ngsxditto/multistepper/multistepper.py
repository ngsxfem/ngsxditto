class MultiStepper:
    """
    This class allows handling multiple steps at the same time and checks while automatically applying redistancing.
    """
    def __init__(self):
        self.transport = None
        self.fluid = None

    def SetTransport(self, transport):
        self.transport = transport

    def RunFixedSteps(self, n):
        for _ in range(n):
            self.transport.OneStep()

    def RunUntilTime(self, end_time):
        if self.transport.time is not None:
            while self.transport.time < end_time:
                self.transport.OneStep()
        else:
            raise TypeError("The transport object has no time parameter")

