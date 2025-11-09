from alive_progress import alive_bar

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
        Applies the OneStep function of the object a given number of times.
        """
        with alive_bar(n, force_tty=True, title="Time stepping: ", bar='smooth') as bar:
            for _ in range(n):
                self.object.Step()
                bar()

    def RunUntilTime(self, end_time):
        """
        Applies the OneStep function of the object until the given time is reached.
        """
        if self.object.time is not None:
            start_time = self.object.time.Get()
            with alive_bar(manual=True, force_tty=True, title="Time stepping: ", bar='smooth') as bar:
                while self.object.time.Get() < end_time:
                    self.object.Step()
                    bar((self.object.time.Get()-start_time)/(end_time-start_time))

        else:
            raise TypeError("The object has no time parameter")

