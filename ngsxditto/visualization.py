from matplotlib import pyplot as plt
import ngsolve.webgui as ngw
from ngsolve import *


class Visualization:
    def __init__(self):
        pass

    def Initialize(self):
        pass

    def AddData(self):
        pass

    def Visualize(self):
        pass



class SphericityDiagram(Visualization):
    def __init__(self, lset, time):
        super().__init__()
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


class Animation(Visualization):
    def __init__(self, lset, fluid, time, end_time):
        super().__init__()
        self.lset = lset
        self.fluid = fluid
        self.time = time
        self.gf_vis = None
        self.gf_vis_tmp = None
        self.vis_last_time = None
        self.vis_time_increment = None
        self.end_time = end_time

    def Initialize(self):
        self.gf_vis = GridFunction(L2(mesh=self.fluid.mesh, order=self.fluid.V.globalorder + 1, dim=4), multidim=0)
        self.gf_vis_tmp = GridFunction(L2(mesh=self.fluid.mesh, order=self.fluid.V.globalorder + 1, dim=4))
        self.vis_last_time = self.time.Get()
        self.vis_time_increment = (self.end_time - self.vis_last_time) / 16

    def AddData(self):
        if self.time.Get() >= self.vis_last_time + self.vis_time_increment:
            self.vis_last_time = self.time.Get()
            self.gf_vis_tmp.Set(
                CF((self.lset.field, Norm(CF((self.fluid.gfu.components[0]))), -1, 0)))
            self.gf_vis.AddMultiDimComponent(self.gf_vis_tmp.vec)

    def Draw(self):
        ngw.Draw(self.gf_vis, self.fluid.mesh, "uhnorm", eval_function="value.x>0.0?value.z:value.y", autoscale=False, min=-0.075,
                 max=0.225, interpolate_multidim=True, animate=True)
