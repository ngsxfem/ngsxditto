from ngsolve import *

from .basetransport import BaseTransport

import typing 

# taken and adapted from NGSolve's modeltemplates

class ExplicitDGTransport(BaseTransport):
    
    def __init__(self, mesh, wind, inflow_values, timestepsize, order=2, source=None, usetrace=True, compile=True):
        super().__init__(mesh, wind, inflow_values, timestepsize, source)
        
        fes = L2(mesh, order=order, all_dofs_together=True)
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
        self.tempu = self.bfa.mat.CreateColVector()
    
    def SetInitialValues(self, initial_values: CoefficientFunction, initial_time: float = 0.0):
        if self.time is not None:
            self.time.Set(initial_time)
        self.gfu.Set (initial_values)

    def Propagate(self, t_old: float, t_new: float):
        n = (t_new - t_old) / self.timestepsize
        if (n - round(n)) > 1e-6:
            raise Exception("timesteps not aligned - adaptivity not implemented")
        for i in range(round(n)):
            if self.time is not None:
                self.time.Set(t_old + i * self.timestepsize)
            self.tempu.data = self.gfu.vec - 0.5 * self.timestepsize * self.invMA * self.gfu.vec
            if self.time is not None:
                self.time.Set(t_old + (i+0.5) * self.timestepsize)
            self.gfu.vec.data -= self.timestepsize * self.invMA * self.tempu
            for callback in self.callbacks:
                callback()

    @property
    def field(self):
        return self.gfu