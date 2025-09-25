from ngsolve import CoefficientFunction, Parameter
from alive_progress import alive_bar
from ngsxditto.stepper import *
import typing


class ProgressInfo:
    def __init__(self):
        pass

    def GetProgressInfo(self):
        raise NotImplementedError()

    def Increment(self):
        raise NotImplementedError()

class DummyProgressInfo(ProgressInfo):
    def __init__(self):
        super().__init__()
        pass

    def GetProgressInfo(self):
        return 0

    def Increment(self):
        pass

class TimeProgressInfo(ProgressInfo):
    def __init__(self, time:Parameter, end_time: float, dt:float):
        super().__init__()
        self.time = time
        self.start_time = self.time.Get()
        self.end_time = end_time
        self.dt = dt

    def GetProgressInfo(self):
        return (self.time.Get() - self.start_time) / (self.end_time - self.start_time)

    def Increment(self):
        self.time.Set(self.time.Get() + self.dt)


class Solver:
    """
    A solver class that registers functions and loops over them when called.
    """
    def __init__(self, stopping_rule: typing.Callable[[], bool] = None,
                 progress_info: ProgressInfo = DummyProgressInfo,
                 should_finalize: typing.Callable[[], bool] = None
                 ):
        """
        Initialize the solver with empty function dictionary.

        Parameters:
        -----------
        stopping_rule: typing.Callable[[], bool]
            Determines when to stop the loop.
        progress_info: typing.Callable[[], float]
            Determines what the progress bar shows
        """
        self.name = "Solver"
        self.stepper_dict = {}
        self.stepper_names = []
        self.stopping_rule = stopping_rule
        self.progress_info = progress_info
        self.i_outer = 0
        self.i_inner = 0
        if should_finalize is None:
            def should_finalize():
                return True
        self.should_finalize = should_finalize


    def SetFinalizeRule(self, should_finalize:typing.Callable[[], bool]):
        self.should_finalize = should_finalize


    def Register(self, stepper_object, name: str=None, step_frequency: int=None, time_frequency: float=None):
        """
        Registers a function with arguments that wil be called in the loop.
        Parameters:
        -----------
        func: function
            The function that will be called
        args: tuple
            The arguments of the function
        name: str
            The name of the call that will be saved in the function name list.
        step_frequency: int
            The function will be called every `step_frequency` steps.
        time_frequency: float
            The function will always be called the first time a new multiple of `time_frequency` is exceeded.
        """

        if name is None:
            name = "unnamed_call_" + str(len(self.stepper_names))

        if name in self.stepper_names:
            raise ValueError(f"Function name {name} already exists.")

        self.stepper_dict[name] = {"object": stepper_object,
                                    "step_frequency": step_frequency,
                                    "time_frequency": time_frequency,
                                    "last_time": 0}
        self.stepper_names.append(name)


    def BeforeLoop(self):
        """
        Will be called before the loop.
        """
        pass

    def AfterLoop(self):
        """
        Will be called after the loop.
        """
        pass

    def __call__(self):
        """
        Executes all function calls that were registered.
        """
        for stepper_name in self.stepper_names:
            stepper_object = self.stepper_dict[stepper_name]["object"]
            stepper_object.BeforeLoop()

        with alive_bar(manual=True, force_tty=True, title=self.name+": ",
                       bar='smooth') as bar:
            while True:
                for stepper_name in self.stepper_names:
                    bar.text = "Current step: " + stepper_name

                    entry = self.stepper_dict[stepper_name]
                    stepper_object = entry["object"]
                    step_frequency = entry["step_frequency"]
                    time_frequency = entry["time_frequency"]
                    should_run = False

                    if step_frequency is not None:
                        should_run = ((self.i_outer+1) % step_frequency == 0)

                    elif time_frequency is not None and hasattr(self, "time"):
                        last_time = entry["last_time"]
                        if int(self.time.Get() // time_frequency) > int(last_time // time_frequency):
                            entry["last_time"] = self.time.Get()
                            should_run = True
                    else:
                        should_run = True

                    if should_run:
                        stepper_object.Step()

                self.i_inner += 1

                if self.should_finalize():
                    for stepper_name in self.stepper_names:
                        stepper_object = self.stepper_dict[stepper_name]["object"]
                        stepper_object.ValidateState()
                    self.i_outer += 1
                    self.i_inner = 0
                    self.progress_info.Increment()
                    bar(self.progress_info.GetProgressInfo())

                else:
                    for stepper_name in self.stepper_names:
                        stepper_object = self.stepper_dict[stepper_name]["object"]
                        stepper_object.RevertState()
                if self.stopping_rule():
                    break

            for stepper_name in self.stepper_names:
                stepper_object = self.stepper_dict[stepper_name]["object"]
                stepper_object.AfterLoop()


class TimeLoop(Solver):
    """
    A Solver subclass that tracks progress with a time parameter.
    """
    def __init__(self, time : typing.Optional[CoefficientFunction] = None, 
                 dt : float = 0.1,
                 end_time : float = 1.0,
                 should_finalize: typing.Callable[[], bool] = None):
        """
        Initialize the timeloop with a time parameter, step-size and end time.
        Parameters:
        -----------
        time: CoefficientFunction
            The time object that is increased every step.
        dt: float
            The time-step size
        end_time: float
            Time when the loop is stopped.
        """
        self.name = "Time Loop"
        if time is None:
            self.time = Parameter(0)
        else:
            self.time = time 
        self.end_time = end_time
        self.start_time = self.time.Get()
        self.dt = dt

        def reached_final_time():
            return self.time.Get() >= self.end_time - 0.1*self.dt


        progress_info = TimeProgressInfo(self.time, self.end_time, dt)
        super().__init__(stopping_rule=reached_final_time, progress_info=progress_info, should_finalize=should_finalize)



from time import sleep
if __name__ == "__main__":

    list_i = [3]

    def increase(list_i):
        list_i[0] += 1
    
    def print_i():
        print(list_i[0])


    def i_too_large():
        return list_i[0] >= 10

    def sleep1():
        sleep(0.1)

    def progress_i():
        return list_i[0]/10

    solver = Solver(stopping_rule=i_too_large, progress_info=progress_i)
    solver.Register(increase,list_i, name="increase list_i")
    solver.Register(print_i, name="print list_i")
    solver.Register(sleep1, name="sleep 1")

    solver()

    t = Parameter(0)
    tl = TimeLoop(time=t, end_time=10)
    def print_time():
        print("time is: " + str(tl.time.Get()))
    tl.Register(print_time, name="print time")
    tl.Register(sleep1, name="sleep 1")

    def increase_dt():
        tl.dt = tl.dt * 1.1

    tl.Register(increase_dt, name="increase dt")

    tl.dt = 0.5
    tl()
