from ngsolve import CoefficientFunction, Parameter, TaskManager
from alive_progress import alive_bar
from ngsxditto.stepper import *
from ngsxditto.progress_info import *
import typing
import time
from contextlib import contextmanager


class Solver:
    """
    A solver class that registers functions and loops over them when called.
    """
    def __init__(self, stopping_rule: typing.Callable[[], bool] = None,
                 progress_info: ProgressInfo = DummyProgressInfo(),
                 should_finalize: typing.Callable[[], bool] = None,
                 should_revert: typing.Callable[[], bool] = None,
                 display_progress_bar:bool=True,
                 show_profiles:bool=True,
                 pajetrace=0
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
        if should_revert is None:
            def should_revert():
                return False
        self.should_finalize = should_finalize
        self.should_revert = should_revert
        self.show_profiles = show_profiles
        self.pajetrace = pajetrace

        @contextmanager
        def dummy_bar(*args, **kwargs):
            yield lambda x=None: None

        self.progress_bar = alive_bar if display_progress_bar else dummy_bar


    def SetFinalizeRule(self, should_finalize:typing.Callable[[], bool]):
        self.should_finalize = should_finalize

    def SetRevertRule(self, should_revert:typing.Callable[[], bool]):
        self.should_revert = should_revert


    def Register(self, stepper_object, name: str=None, step_frequency: int=None, time_frequency: float=None, as_validate:bool=False, measure_time=None):
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
            stepper_object = FunctionCallStepper(stepper_object, as_validate=as_validate)

        if measure_time is not None:
            if measure_time:
                stepper_object.auto_time = True
            if not measure_time:
                stepper_object.auto_time = False

        self.stepper_dict[name] = {"object": stepper_object,
                                   "step_frequency": step_frequency,
                                   "time_frequency": time_frequency,
                                   "next_trigger": 0}
        self.stepper_names.append(name)
        stepper_object._solver = self


    def __call__(self):
        """
        Executes all 'Step' functions of the objects that were registered.
        """
        should_run_dict = {}
        for stepper_name in self.stepper_names:
            entry = self.stepper_dict[stepper_name]
            stepper_object = entry["object"]
            stepper_object.reset_times()
            stepper_object.BeforeLoop()
            if entry["time_frequency"] is not None:
                entry["next_trigger"] += entry["time_frequency"]
            should_run_dict[stepper_name] = True

        with self.progress_bar(manual=True, force_tty=True, title=self.name+": ",
                       bar='smooth') as bar:
            with TaskManager(pajetrace=self.pajetrace):
                while True:
                    self.progress_info.Step()

                    for stepper_name in self.stepper_names:
                        bar.text = "Current step: " + stepper_name

                        entry = self.stepper_dict[stepper_name]

                        should_run = False
                        if entry["step_frequency"] is not None:
                            should_run = ((self.i_outer + 1) % entry["step_frequency"] == 0)

                        elif entry["time_frequency"] is not None and hasattr(self, "time"):
                            if self.time.Get() + 1e-8 >= entry["next_trigger"]:
                                should_run = True

                        else:
                            should_run = True

                        should_run_dict[stepper_name] = should_run

                    for stepper_name in self.stepper_names:
                        entry = self.stepper_dict[stepper_name]
                        stepper_object = entry["object"]
                        if should_run_dict[stepper_name]:
                            stepper_object.Step()

                    self.i_inner += 1

                    if self.should_finalize():
                        #self.progress_info.Increment()
                        self.progress_info.ValidateStep()
                        for stepper_name in self.stepper_names:
                            entry = self.stepper_dict[stepper_name]
                            stepper_object = entry["object"]

                            if should_run_dict[stepper_name]:
                                stepper_object.ValidateStep()
                                if entry["time_frequency"] is not None:
                                    entry["next_trigger"] += entry["time_frequency"]

                        self.i_outer += 1
                        self.i_inner = 0
                        bar(self.progress_info.GetProgressInfo())

                    elif self.should_revert():
                        self.progress_info.RevertStep()
                        for stepper_name in self.stepper_names:
                            entry = self.stepper_dict[stepper_name]
                            stepper_object = entry["object"]

                            if should_run_dict[stepper_name]:
                                stepper_object.RevertStep()
                        self.i_outer += 1
                        self.i_inner = 0
                        bar(self.progress_info.GetProgressInfo())

                    else:
                        self.progress_info.AcceptIntermediate()
                        for stepper_name in self.stepper_names:
                            entry = self.stepper_dict[stepper_name]
                            stepper_object = entry["object"]
                            if should_run_dict[stepper_name]:
                                stepper_object.AcceptIntermediate()

                    if self.stopping_rule():
                        break

        for stepper_name in self.stepper_names:
            entry = self.stepper_dict[stepper_name]
            stepper_object = entry["object"]
            stepper_object.AfterLoop()
        if self.show_profiles:
            for stepper_name in self.stepper_names:
                if self.stepper_dict[stepper_name]["object"].auto_time:
                    entry = self.stepper_dict[stepper_name]
                    time_dict = entry["object"].times
                    print(f"{stepper_name}: {time_dict['__total__']}")
                    if len(entry["object"].times) > 1:
                        for key, value in time_dict.items():
                            if key != "__total__":
                                print(f"    {key}: {value}")



class TimeLoop(Solver):
    """
    A Solver subclass that tracks progress with a time parameter.
    """
    def __init__(self, time : typing.Optional[CoefficientFunction] = None, 
                 dt : float = 0.1,
                 end_time : float = 1.0,
                 should_finalize: typing.Callable[[], bool] = None,
                 should_revert: typing.Callable[[], bool] = None,
                 display_progress_bar:bool=True,
                 show_profiles: bool = True,
                 pajetrace: int = 0
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
                         should_finalize=should_finalize, should_revert=should_revert, display_progress_bar=display_progress_bar,
                         show_profiles=show_profiles, pajetrace=pajetrace)

    def SetTimeStepSize(self, dt):
        self.dt = dt
        self.progress_info.SetTimeStepSize(dt)
