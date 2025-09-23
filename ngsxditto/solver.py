from ngsolve import CoefficientFunction, Parameter
from alive_progress import alive_bar
from ngsxditto.stateholder import *
import typing



def dummy_progress_info():
    return 0

class Solver:
    """
    A solver class that registers functions and loops over them when called.
    """
    def __init__(self, stopping_rule: typing.Callable[[], bool] = None,
                 progress_info: typing.Callable[[], float] = dummy_progress_info):
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
        self.function_dict = {}
        self.function_fin_dict = {}
        self.function_names = []        
        self.stopping_rule = stopping_rule
        self.progress_info = progress_info
        self.visualizations = []

    def AddVisualization(self, visualization):
        """
        Adds a Visualization object to the list of visualizations.
        """
        self.visualizations.append(visualization)

    def Register(self, func, *args: typing.Any, name: str=None, step_frequency: int=None, time_frequency: float=None):
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
            name = "unnamed_call_" + str(len(self.function_names))

        if name in self.function_names:
            raise ValueError(f"Function name {name} already exists.")

        func_of_class = getattr(func, "__func__", func)  # func is a bound method, func_of_class is a plain function

        base_func = getattr(Stateholder, "Step", None)

        if func_of_class is base_func:
            self.function_fin_dict[name] = func.__self__.StoreState
        #if not callable(func_call):
        #    raise ValueError(f"Function {name} is not callable.")

        self.function_dict[name] = {"call": (func,args),
                                    "step_frequency": step_frequency,
                                    "time_frequency": time_frequency,
                                    "last_time": 0}
        self.function_names.append(name)


    def BeforeLoop(self):
        """
        Will be called before the loop.
        """
        for vis in self.visualizations:
            vis.Initialize()
            self.Register(vis.AddData, name=vis.name, step_frequency=vis.step_frequency, time_frequency=vis.time_frequency)


    def AfterLoop(self):
        """
        Will be called after the loop.
        """
        for vis in self.visualizations:
            vis.Draw()

    def __call__(self):
        """
        Executes all function calls that were registered.
        """
        self.BeforeLoop()
        with alive_bar(manual=True, force_tty=True, title=self.name+": ",
                       bar='smooth') as bar:
            i = 1
            while True:
                for function_name in self.function_names:
                    bar.text = "Current step: " + function_name

                    entry = self.function_dict[function_name]
                    func, args = entry["call"]
                    step_frequency = entry["step_frequency"]
                    time_frequency = entry["time_frequency"]
                    should_run = False

                    if step_frequency is not None:
                        should_run = (i % step_frequency == 0)

                    elif time_frequency is not None and hasattr(self, "time"):
                        last_time = entry["last_time"]
                        if int(self.time.Get() // time_frequency) > int(last_time // time_frequency):
                            entry["last_time"] = self.time.Get()
                            should_run = True
                    else:
                        should_run = True

                    if should_run:
                        func(*args)

                bar(self.progress_info())

                if self.stopping_rule():
                    break
                i += 1
            self.AfterLoop()


class TimeLoop(Solver):
    """
    A Solver subclass that tracks progress with a time parameter.
    """
    def __init__(self, time : typing.Optional[CoefficientFunction] = None, 
                 dt : float = 0.1,
                 end_time : float = 1.0):
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
        super().__init__()
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

        def relative_time():
            return (self.time.Get()-self.start_time)/(self.end_time-self.start_time)

        self.stopping_rule = reached_final_time
        self.progress_info = relative_time


    def BeforeLoop(self):
        super().BeforeLoop()
        def finalizestates():
            for function_name in self.function_names:
                if function_name in self.function_fin_dict:
                    self.function_fin_dict[function_name]()
        self.Register(finalizestates, name="finalize states")

        time_increase = lambda: self.time.Set(self.time.Get() + self.dt)
        self.Register(time_increase, name="increase time value")



class TimeLoop2(TimeLoop):
    """
    A Solver subclass that tracks progress with a time parameter.
    """
    def __init__(self, time : typing.Optional[CoefficientFunction] = None, 
                 dt : float = 0.1,
                 end_time : float = 1.0,
                 should_finalize: typing.Callable[[], bool]=None):
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
        super().__init__(time,dt,end_time)
        if should_finalize is None:
            def should_finalize():
                return self.countits == 10
        self.should_finalize = should_finalize
        self.countits = 0


    def CheckIteration(self):
        self.countits += 1
        if self.should_finalize():
            for function_name in self.function_names:
                if function_name in self.function_fin_dict:
                    self.function_fin_dict[function_name]()

            time_increase = lambda: self.time.Set(self.time.Get() + self.dt)
            time_increase()
            print("time increased: ", self.time.Get())
            self.countits = 0


    def BeforeLoop(self):
        super(TimeLoop,self).BeforeLoop()
        self.Register(self.CheckIteration, name="CheckIteration")




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
