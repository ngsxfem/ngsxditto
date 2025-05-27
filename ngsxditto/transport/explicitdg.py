from ngsolve import *

from .basetransport import BaseTransport

import typing 

# taken and adapted from NGSolve's modeltemplates

class ExplicitDGTransport(BaseTransport):
    
    def __init__(self, mesh, wind, inflow_values, dt, order=2, source=None, usetrace=True, compile=True):
        super().__init__(mesh, wind, inflow_values, dt, source, order=order)
        
        fes = L2(mesh, order=order, all_dofs_together=True)
        fes_cont = H1(mesh, order=order)
        u,v = fes.TnT()
        
        wind = self.wind

        if not usetrace:
            self.bfa = BilinearForm(fes, nonassemble=True)
            self.bfa += -u * wind * grad(v) * dx
            wn = wind*specialcf.normal(mesh.dim)
            self.bfa += (wn * IfPos(wn, u, u.Other(bnd=inflow_values)) * v).Compile(compile, wait=True) * dx(element_boundary=True)
            aop = self.bfa.mat
        else:
            fes_trace = Discontinuous(FacetFESpace(mesh, order=order))
            utr,vtr = fes_trace.TnT()
            trace = fes.TraceOperator(fes_trace, False)
            
            self.bfa = BilinearForm(fes, nonassemble=True)
            self.bfa += -u * wind * grad(v) * dx

            self.bfa_trace = BilinearForm(fes_trace, nonassemble=True)
            wn = wind*specialcf.normal(mesh.dim)
            self.bfa_trace += (wn * IfPos(wn, utr, utr.Other(bnd=inflow_values)) * vtr).Compile(compile,wait=True) * dx(element_boundary=True)

            aop = self.bfa.mat + trace.T @ self.bfa_trace.mat @ trace
        
        self.invmass = fes.Mass(rho=1).Inverse()
        self.invMA = self.invmass @ aop
        self.gfu = GridFunction(fes)
        self.gfu_cont = GridFunction(fes_cont)
        self.tempu = self.bfa.mat.CreateColVector()
    
    def SetInitialValues(self, initial_values: CoefficientFunction, initial_time: float = 0.0):
        if self.time is not None:
            self.time.Set(initial_time)
        self.gfu.Set (initial_values)
        self.gfu_cont.Set(self.gfu)

    def SetWind(self, wind: CoefficientFunction):
        raise NotImplementedError("SetWind not yet implemented")

    def OneStep(self):
        self.tempu.data = self.gfu.vec - 0.5 * self.dt * self.invMA * self.gfu.vec
        if self.time is not None:
            self.time.Set(self.time.Get() + self.dt)
        self.gfu.vec.data -= self.dt * self.invMA * self.tempu
        self.gfu_cont.Set (self.gfu)


    def Propagate(self, t_old: float, t_new: float):
        n = (t_new - t_old) / self.dt
        if (n - round(n)) > 1e-6:
            raise Exception("timesteps not aligned - adaptivity not implemented")
        for i in range(round(n)):
            if self.time is not None:
                self.time.Set(t_old + i * self.dt)
            self.tempu.data = self.gfu.vec - 0.5 * self.dt * self.invMA * self.gfu.vec
            if self.time is not None:
                self.time.Set(t_old + (i+0.5) * self.dt)
            self.gfu.vec.data -= self.dt * self.invMA * self.tempu
        self.gfu_cont.Set(self.gfu)
        # callback inside or outside?
        for callback in self.callbacks:
            callback()

    @property
    def field(self):
        return self.gfu