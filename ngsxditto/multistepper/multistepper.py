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

    def _advance_clock(self):
        """Advance the object's time parameter by its step size (if any).
        Should be called only by the driving loop.
        """
        time = getattr(self.object, "time", None)
        if time is None:
            return
        dt = getattr(self.object, "dt", None)
        if dt is None:
            transport = getattr(self.object, "transport", None)
            dt = getattr(transport, "dt", None)
        if dt is not None:
            time.Set(time.Get() + dt)

    def RunFixedSteps(self, n):
        """
        Applies the OneStep function of the object a given number of times.
        """
        with alive_bar(n, force_tty=True, title="Time stepping: ", bar='smooth') as bar:
            for _ in range(n):
                self.object.Step()
                self.object.ValidateStep()
                self._advance_clock()
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
                    self.object.ValidateStep()
                    self._advance_clock()
                    bar((self.object.time.Get()-start_time)/(end_time-start_time))

        else:
            raise TypeError("The object has no time parameter")

