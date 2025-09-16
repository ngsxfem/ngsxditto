from matplotlib import pyplot as plt
import ngsolve.webgui as ngw
from ngsolve import *


class Visualization:
    def __init__(self, name=None, step_frequency=None, time_frequency=None):
        self.initialize_funcs = []
        self.adddata_funcs = []
        self.draw_funcs = []
        self.name = name
        self.step_frequency = step_frequency
        self.time_frequency = time_frequency

    def RegisterInitialize(self, func, *args):
        self.initialize_funcs.append((func, args))

    def RegisterAddData(self, func, *args):
        self.adddata_funcs.append((func, args))

    def RegisterDraw(self, func, *args):
        self.draw_funcs.append((func, args))

    def Initialize(self):
        for func, args in self.initialize_funcs:
            func(*args)

    def AddData(self):
        for func, args in self.adddata_funcs:
            func(*args)

    def Draw(self):
        for func, args in self.draw_funcs:
            func(*args)



class SphericityDiagram(Visualization):
    def __init__(self, lset, time, name=None, step_frequency=None, time_frequency=None):
        super().__init__(name, step_frequency, time_frequency)
        self.lset = lset
        self.time = time
        self.time_list = []
        self.surface_volume_ratio = []

    def Initialize(self):
        self.time_list = [self.time.Get()]
        self.surface_volume_ratio = [self.lset.surface_area / self.lset.volume]

    def AddData(self):
        self.time_list.append(self.time.Get())
        self.surface_volume_ratio.append(self.lset.surface_area/self.lset.volume)

    def Draw(self):
        plt.plot(self.time_list, self.surface_volume_ratio)
        plt.show()


class UnfittedNGSWebguiPlot(Visualization):
    def __init__(self, lset, cf_neg, cf_pos, order, time, end_time, name=None, step_frequency=None, time_frequency=None,
                 min=0, max=1, autoscale=True):
        super().__init__(name, step_frequency, time_frequency)
        self.lset = lset
        self.cf_neg = cf_neg
        self.cf_pos = cf_pos
        self.order = order
        self.time = time
        self.gf_vis = None
        self.gf_vis_tmp = None
        self.vis_last_time = None
        self.vis_time_increment = None
        self.end_time = end_time
        self.min = min
        self.max = max
        self.autoscale = autoscale

    def Initialize(self):
        self.gf_vis = GridFunction(L2(mesh=self.lset.mesh, order=self.order + 1, dim=4), multidim=0)
        self.gf_vis_tmp = GridFunction(L2(mesh=self.lset.mesh, order=self.order + 1, dim=4))
        self.vis_last_time = self.time.Get()
        self.vis_time_increment = (self.end_time - self.vis_last_time) / 16

    def AddData(self):
        if self.time.Get() >= self.vis_last_time + self.vis_time_increment:
            self.vis_last_time = self.time.Get()
            self.gf_vis_tmp.Set(
                CF((self.lset.field, self.cf_neg, self.cf_pos, 0)))
            self.gf_vis.AddMultiDimComponent(self.gf_vis_tmp.vec)

    def Draw(self):
        ngw.Draw(self.gf_vis, self.lset.mesh, "uhnorm", eval_function="value.x>0.0?value.z:value.y",
                 autoscale=self.autoscale, min=self.min, max=self.max, interpolate_multidim=True, animate=True)

