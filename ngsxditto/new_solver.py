from ngsolve import CoefficientFunction, Parameter
from alive_progress import alive_bar

import typing 

def dummy_progress_info():
    return 0

class Solver:
    def __init__(self, stopping_rule = None, progress_info=dummy_progress_info):
        self.name = "Solver"
        self.function_dict = {}
        self.function_names = []        
        self.stopping_rule = stopping_rule
        self.progress_info = progress_info

    def Register(self, func, *args, name=None):

        if name is None:
            name = "unnamed_call_" + str(len(self.function_names))

        if name in self.function_names:
            raise ValueError(f"Function name {name} already exists.")

        self.function_dict[name] = (func,args)
        self.function_names.append(name)

    def BeforeLoop(self):
        pass

    def __call__(self):
        self.BeforeLoop()
        with alive_bar(manual=True, force_tty=True, title=self.name+": ", 
                       bar='smooth') as bar:
            while True:
                for function_name in self.function_names:
                    bar.text = "Current step: " + function_name
                    func, args = self.function_dict[function_name]

                    func(*args)
                bar(self.progress_info())

                if self.stopping_rule():
                    break


class TimeLoop(Solver):
    def __init__(self, time : typing.Optional[CoefficientFunction] = None, 
                 dt : float = 0.1,
                 end_time : float = 1.0):
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
        time_increase = lambda: self.time.Set(self.time.Get() + self.dt)
        self.Register(time_increase, name="increase time value")

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
