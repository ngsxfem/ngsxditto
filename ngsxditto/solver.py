from ngsolve import CoefficientFunction, Parameter, TaskManager
from alive_progress import alive_bar
from ngsxditto.stepper import *
import typing
import time
from contextlib import contextmanager



class ProgressInfo:
    """
    This class keeps track of the progress.
    """
    def __init__(self):
        pass

    def GetProgressInfo(self):
        raise NotImplementedError("GetProgressInfo not implemented in base class")

    def Increment(self):
        raise NotImplementedError("Increment not implemented in base class")

class DummyProgressInfo(ProgressInfo):
    """
    Always returns 0 and can not be incremented.
    """
    def __init__(self):
        super().__init__()
        pass

    def GetProgressInfo(self):
        return 0

    def Increment(self):
        pass

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
        self.start_time = self.time.Get()
        self.end_time = end_time
        self.dt = dt

    def GetProgressInfo(self):
        """
        Returns:
        --------
        float:
            The progress info defined by where the time parameter lies between start and end time.
        """
        return (self.time.Get() - self.start_time) / (self.end_time - self.start_time)

    def Increment(self):
        """
        Increase the time parameter by dt.
        """
        self.time.Set(self.time.Get() + self.dt)

    def SetTimeStepSize(self, dt):
        self.dt = dt


class IterationProgressInfo(ProgressInfo):
    def __init__(self, n_end: int=10, n_start: int=0):
        super().__init__()
        self.n = self.n_start =  n_start
        self.n_end = n_end

    def GetProgressInfo(self):
        return (self.n - self.n_start) / (self.n_end - self.n_start)

    def Increment(self):
        self.n += 1


class Solver:
    """
    A solver class that registers functions and loops over them when called.
    """
    def __init__(self, stopping_rule: typing.Callable[[], bool] = None,
                 progress_info: ProgressInfo = DummyProgressInfo(),
                 should_finalize: typing.Callable[[], bool] = None,
                 display_progress_bar:bool=True,
                 show_profiles:bool=True
                 ):
        """
        Initialize the solver with empty dictionary that can be filled with Stepper objects.

        Parameters:
        -----------
        stopping_rule: typing.Callable[[], bool]
            Determines when to stop the loop.
        progress_info: ProgressInfo
            Determines what the progress bar shows
        should_finalize: typing.Callable[[], bool]
            The criteria to determine if the solver should finalize the step.
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
        self.show_profiles = show_profiles

        @contextmanager
        def dummy_bar(*args, **kwargs):
            yield lambda x=None: None

        self.progress_bar = alive_bar if display_progress_bar else dummy_bar


    def SetFinalizeRule(self, should_finalize:typing.Callable[[], bool]):
        self.should_finalize = should_finalize


    def Register(self, stepper_object, name: str=None, step_frequency: int=None, time_frequency: float=None, validate_only:bool=False):
        """
        Registers a function with arguments that wil be called in the loop.
        Parameters:
        -----------
        stepper_object: Stepper
            Determines what function will be called in the loop.
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

        if callable(stepper_object):
            stepper_object = FunctionCallStepper(stepper_object, validate_only=validate_only)

        self.stepper_dict[name] = {"object": stepper_object,
                                   "step_frequency": step_frequency,
                                   "time_frequency": time_frequency,
                                   "total_computation_time": 0,
                                   "last_time": 0}
        self.stepper_names.append(name)


    def __call__(self):
        """
        Executes all 'Step' functions of the objects that were registered.
        """
        for stepper_name in self.stepper_names:
            entry = self.stepper_dict[stepper_name]
            stepper_object =entry["object"]
            start_time = time.time()
            stepper_object.BeforeLoop()
            end_time = time.time()
            entry["total_computation_time"] += (end_time - start_time)

        with self.progress_bar(manual=True, force_tty=True, title=self.name+": ",
                       bar='smooth') as bar:
            with TaskManager():
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
                            start_time = time.time()
                            stepper_object.Step()
                            end_time = time.time()
                            entry["total_computation_time"] += (end_time - start_time)

                    self.i_inner += 1

                    if self.should_finalize():
                        self.progress_info.Increment()
                        for stepper_name in self.stepper_names:
                            entry = self.stepper_dict[stepper_name]
                            stepper_object = entry["object"]
                            step_frequency = entry["step_frequency"]
                            time_frequency = entry["time_frequency"]

                            should_run = False

                            if step_frequency is not None:
                                should_run = ((self.i_outer + 1) % step_frequency == 0)

                            elif time_frequency is not None and hasattr(self, "time"):
                                last_time = entry["last_time"]
                                if int(self.time.Get() // time_frequency) > int(last_time // time_frequency):
                                    entry["last_time"] = self.time.Get()
                                    should_run = True
                            else:
                                should_run = True
                            if should_run:

                                start_time = time.time()
                                stepper_object.ValidateStep()
                                end_time = time.time()
                                entry["total_computation_time"] += (end_time - start_time)

                        self.i_outer += 1
                        self.i_inner = 0
                        bar(self.progress_info.GetProgressInfo())

                    else:
                        for stepper_name in self.stepper_names:
                            entry = self.stepper_dict[stepper_name]
                            stepper_object = entry["object"]
                            start_time = time.time()
                            stepper_object.RevertStep()
                            end_time = time.time()
                            entry["total_computation_time"] += (end_time - start_time)

                    if self.stopping_rule():
                        break

        for stepper_name in self.stepper_names:
            entry = self.stepper_dict[stepper_name]
            stepper_object = entry["object"]
            start_time = time.time()
            stepper_object.AfterLoop()
            end_time = time.time()
            entry["total_computation_time"] += (end_time - start_time)
        if self.show_profiles:
            for stepper_name in self.stepper_names:
                print(f"{stepper_name}: {self.stepper_dict[stepper_name]['total_computation_time']}")



class TimeLoop(Solver):
    """
    A Solver subclass that tracks progress with a time parameter.
    """
    def __init__(self, time : typing.Optional[CoefficientFunction] = None, 
                 dt : float = 0.1,
                 end_time : float = 1.0,
                 should_finalize: typing.Callable[[], bool] = None,
                 display_progress_bar:bool=True,
                 show_profiles: bool = True
                 ):
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
        should_finalize: typing.Callable[[], bool]
            The criteria to determine if the solver should finalize the step.

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


        self.progress_info = TimeProgressInfo(self.time, self.end_time, self.dt)
        self.show_profiles = show_profiles
        super().__init__(stopping_rule=reached_final_time, progress_info=self.progress_info,
                         should_finalize=should_finalize, display_progress_bar=display_progress_bar,
                         show_profiles=show_profiles)

    def SetTimeStepSize(self, dt):
        self.dt = dt
        self.progress_info.SetTimeStepSize(dt)



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
