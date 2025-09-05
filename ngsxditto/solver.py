from ngsxditto.fluid import FluidDiscretization
from ngsxditto.levelset import LevelSetGeometry
from ngsolve import *
import ngsolve.webgui as ngw
from matplotlib import pyplot as plt

from alive_progress import alive_bar

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
            #gfu_u = GridFunction(self.fluid.V, multidim=0)
            #gfu_u_tmp = GridFunction(self.fluid.V)
            gf_vis = GridFunction(L2(mesh=self.mesh,order=self.fluid.V.globalorder+1,dim=4), multidim=0)
            gf_vis_tmp = GridFunction(L2(mesh=self.mesh,order=self.fluid.V.globalorder+1,dim=4))
            vis_last_time = self.time.Get()
            vis_time_increment = (end_time - vis_last_time)/16

        if round(self.time.Get(), 5) in draw_at_times:
            self.DrawSolution()

        if sphericity_diagram:
            time_list = [self.time.Get()]
            surface_volume_ratio = [self.lset.surface_area/self.lset.volume]


        with alive_bar(manual=True, force_tty=True) as bar:
          while self.time < end_time:
            timeold = self.time.Get()
            for func, info in self.function_dict.items():
                bartxt = "Current step:" 
                if info["name"] is not None:
                    bartxt += info["name"]
                bar.text = bartxt 
                args = info["args"]
                name = info["name"]

                resolved_args = resolve_args(args)

                result = func(*resolved_args)

                if name is not None:
                    self.variable_dict[name] = result


            if round(self.time.Get(), 5) in draw_at_times:
                self.DrawSolution()

            if animate:
                if self.time.Get() >= vis_last_time + vis_time_increment:
                    vis_last_time = self.time.Get()
                    #gfu_u_tmp.Set(IfPos(self.lset.field, CF((0, 0)), self.fluid.gfu.components[0]))
                    gf_vis_tmp.Set(CF((self.lset.field, Norm(CF((self.fluid.gfu.components[0],self.fluid.gfu.components[1]))),-1,0)))
                    #gfu_u.AddMultiDimComponent(gfu_u_tmp.vec)
                    gf_vis.AddMultiDimComponent(gf_vis_tmp.vec)

            if sphericity_diagram:
                time_list.append(self.time.Get())
                surface_volume_ratio.append(self.lset.surface_area / self.lset.volume)
            bar(self.time.Get()/end_time)
        if animate:
            #ngw.Draw(gfu_u, self.mesh, interpolate_multidim=True, animate=True, min=0, autoscale=False)
            ngw.Draw(gf_vis, self.mesh, "uhnorm",eval_function="value.x>0.0?value.z:value.y",autoscale=False, min=-0.075,max=0.225, interpolate_multidim=True, animate=True)

        if sphericity_diagram:
            plt.plot(time_list, surface_volume_ratio)
            plt.show()

    def DrawSolution(self):
        ngw.Draw(IfPos(self.lset.field, CF((0, 0)), self.fluid.gfu.components[0]), self.mesh)




