class MultiStepper:
    """
    This class allows handling multiple steps at the same time and checks while automatically applying redistancing.
    """
    def __init__(self):
        """
        Initializes the MultiStepper class.
        """
        self.object = None

    def SetObject(self, object):
        """
        Sets the object. The object must be of a class that has a OneStep function.
        """
        self.object = object

    def RunFixedSteps(self, n):
        """
        Applies the OneStep function of the object a given number of times..
        """
        for _ in range(n):
            self.object.OneStep()

    def RunUntilTime(self, end_time):
        """
        Applies the OneStep function of the object until the given time is reached.
        """
        if self.object.time is not None:
            while self.object.time < end_time:
                self.object.OneStep()
        else:
            raise TypeError("The transport object has no time parameter")

