from ngsxditto.stepper import *
from ngsolve import Parameter
import typing


class ProgressInfo(StatefulStepper):
    """
    This class keeps track of the progress.
    """
    auto_time = False
    def __init__(self):
        super().__init__()

    def GetProgressInfo(self):
        raise NotImplementedError("GetProgressInfo not implemented in base class")


class DummyProgressInfo(ProgressInfo):
    """
    Always returns 0 and can not be incremented.
    """
    def __init__(self):
        super().__init__()
        pass

    def GetProgressInfo(self):
        return 0

    def Step(self):
        pass

    def RevertStep(self):
        pass

    def AcceptIntermediate(self):
        pass

    def ValidateStep(self):
        pass

    def ComputeDifference2Intermediate(self):
        return 0

class TimeProgressInfo(ProgressInfo):
    """
    In this class the progress is defined by elapsed time.
    """
    def __init__(self, time:Parameter, end_time: float, dt:float):
        """
        Initialize the time progress info.
        Parameters:
        -----------
        time : Parameter
            The parameter that keeps track of the time.
        end_time : float
            The time when the progress is 1 (100%).
        dt : float
            The time-step size
        """
        super().__init__()
        self.time = time
        self.current = self.time
        self.start_time = self.time.Get()
        self.end_time = end_time
        self.dt = dt

        self.past = Parameter(self.start_time)
        self.intermediate = Parameter(self.start_time)

    def GetProgressInfo(self):
        """
        Returns:
        --------
        float:
            The progress info defined by where the time parameter lies between start and end time.
        """
        return (self.time.Get() - self.start_time) / (self.end_time - self.start_time)



    def Step(self):
        self.Increment()

    def ValidateStep(self):
        self.past.Set(self.current.Get())
        self.intermediate.Set(self.current.Get())

    def AcceptIntermediate(self):
        self.intermediate.Set(self.time.Get())
        self.time.Set(self.time.Get() - self.dt)

    def Increment(self):
        """
        Increase the time parameter by dt.
        """
        self.time.Set(self.time.Get() + self.dt)


    def RevertStep(self):
        """
        Decrease the time parameter by dt.
        """
        self.time.Set(self.past.Get())
        self.intermediate.Set(self.past.Get())

    def SetTimeStepSize(self, dt):
        self.dt = dt

    def ComputeDifference2Intermediate(self):
        return abs(self.intermediate.Get() - self.current.Get())


class IterationProgressInfo(ProgressInfo):
    def __init__(self, n_end: int=10, n_start: int=0):
        super().__init__()
        self.n = self.n_start =  n_start
        self.n_end = n_end

    def GetProgressInfo(self):
        return (self.n - self.n_start) / (self.n_end - self.n_start)

    def Step(self):
        self.Increment()

    def Increment(self):
        self.n += 1

    def RevertStep(self):
        self.n -= 1

    def AcceptIntermediate(self):
        self.n -= 1

    def ValidateStep(self):
        pass

    def ComputeDifference2Intermediate(self):
        return 1
