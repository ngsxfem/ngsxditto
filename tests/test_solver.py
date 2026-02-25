from ngsxditto.solver import *


def test_solver_without_inner_loop():
    n_iterations = 10
    progress_info = IterationProgressInfo(n_end=n_iterations)
    solver = Solver(stopping_rule=None, progress_info=progress_info)
    solver.stopping_rule = lambda: solver.i_outer == n_iterations

    n_list = []
    add_first_element = lambda: n_list.append("start")
    fill_list = lambda: n_list.append(solver.i_outer)
    add_last_element = lambda: n_list.append("end")

    fill_list_stepper = FunctionCallStepper(step_function=fill_list, before_loop_function=add_first_element,
                                            after_loop_function=add_last_element)
    solver.Register(fill_list_stepper)

    solver()
    assert solver.i_outer == n_iterations
    assert n_list == ["start"] + [i for i in range(n_iterations)] + ["end"]
    assert solver.progress_info.GetProgressInfo() == 1.


def test_solver_with_inner_loop():
    n_iterations = 10
    progress_info = IterationProgressInfo(n_end=n_iterations)
    solver = Solver(stopping_rule=None, progress_info=progress_info)
    solver.stopping_rule = lambda: solver.i_outer == n_iterations
    solver.should_finalize = lambda: solver.i_inner % 2 == 0

    n_list = []
    add_first_element = lambda: n_list.append("start")
    fill_list = lambda: n_list.append(solver.i_outer)
    add_last_element = lambda: n_list.append("end")

    fill_list_stepper = FunctionCallStepper(step_function=fill_list, before_loop_function=add_first_element,
                                            after_loop_function=add_last_element)
    fill_list_stepper.AcceptIntermediate = lambda: n_list.pop()
    solver.Register(fill_list_stepper)
    solver.Register(lambda: print(solver.i_outer, solver.i_inner))

    solver()
    assert solver.i_outer == n_iterations
    assert n_list == ["start"] + [i for i in range(n_iterations)] + ["end"]
    assert solver.progress_info.GetProgressInfo() == 1.

