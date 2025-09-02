from ngsxditto.fluid import FluidDiscretization
from ngsxditto.levelset import LevelSetGeometry
from ngsolve import *
import ngsolve.webgui as ngw
from matplotlib import pyplot as plt


class Solver:
    def __init__(self, fluid, time=None):
        self.function_dict = {}
        self.variable_dict = {}
        self.time = time
        self.fluid = fluid
        self.lset = self.fluid.lset
        self.mesh = self.fluid.mesh


    def AddObject(self, name, obj):
        setattr(self, name, obj)


    def Add(self, func, *args, name=None):
        self.function_dict[func] = {"name": name, "args": args}

        if name is not None:
            self.variable_dict[name] = func(*args)


    def Do(self, end_time, draw_at_times=[], animate=True, sphericity_diagram=False):
        def resolve_args(args):
            resolved = []
            for arg in args:
                if isinstance(arg, str) and arg in self.variable_dict:
                    resolved.append(self.variable_dict[arg])
                else:
                    resolved.append(arg)
            return tuple(resolved)

        draw_at_times = [round(time_point, 5) for time_point in draw_at_times]
        if animate:
            gfu_u = GridFunction(self.fluid.V, multidim=0)
            gfu_u_tmp = GridFunction(self.fluid.V)

        if round(self.time.Get(), 5) in draw_at_times:
            self.DrawSolution()

        if sphericity_diagram:
            time_list = [self.time.Get()]
            surface_volume_ratio = [self.lset.surface_area/self.lset.volume]

        while self.time < end_time:

            for func, info in self.function_dict.items():
                args = info["args"]
                name = info["name"]

                resolved_args = resolve_args(args)

                result = func(*resolved_args)

                if name is not None:
                    self.variable_dict[name] = result


            if round(self.time.Get(), 5) in draw_at_times:
                self.DrawSolution()

            if animate:
                gfu_u_tmp.Set(IfPos(self.lset.field, CF((0, 0)), self.fluid.gfu.components[0]))
                gfu_u.AddMultiDimComponent(gfu_u_tmp.vec)

            if sphericity_diagram:
                time_list.append(self.time.Get())
                surface_volume_ratio.append(self.lset.surface_area / self.lset.volume)

        if animate:
            ngw.Draw(gfu_u, self.mesh, interpolate_multidim=True, animate=True, min=0, autoscale=False)

        if sphericity_diagram:
            plt.plot(time_list, surface_volume_ratio)
            plt.show()

    def DrawSolution(self):
        ngw.Draw(IfPos(self.lset.field, CF((0, 0)), self.fluid.gfu.components[0]), self.mesh)




