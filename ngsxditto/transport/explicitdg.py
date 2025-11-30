from ngsolve import *
from .basetransport import BaseTransport
from ngsxditto import direct_solver_spd, direct_solver_nonspd
from xfem.utils import *
from xfem import *
import typing

# taken and adapted from NGSolve's modeltemplates

class ExplicitDGTransport(BaseTransport):
    """
    This class propagates a function along a given velocity field (wind) using the Runge-Kutta-2 discretization
    in time and the DG or HDG method as space discretization.
    """
    def __init__(self, mesh, wind=None, inflow_values=None, dt=0.01, order: int=2, source=None, usetrace: bool=True,
                 compile=False, active_elements: typing.Optional[BitArray] = None):
        """
        Initializes the transport object with the given parameters.

        Parameters:
        -----------
        mesh: Mesh
            The computational Mesh
        wind: CoefficientFunction
            The velocity field that transports the levelset
        inflow_values: CoefficientFunction
            The inflow boundary data
        dt: float
            The time step size for the transport.
        order: int
            The order of the discretization.
        usetrace: bool
            If True use a HDG discretization in space. If false use a DG discretization.
        """
        if usetrace and active_elements is not None:
            raise NotImplementedError("Narrow band transport not yet implemented for HDG methods. Set usetrace=False or active_elements=None")

        super().__init__(mesh, wind, inflow_values, dt, source, order=order, active_elements=active_elements)

        self.usetrace = usetrace
        self.compile = compile
        self.fes = L2(mesh, order=order, all_dofs_together=True, dgjumps=True)
        self.u, self.v = self.fes.TnT()
        self.bfa = None
        self.mass = None
        self.invmass = None
        self.invMA = None
        self.tempu = None


        self.gfu = GridFunction(self.fes)
        self.current = self.gfu
        self.past = GridFunction(self.gfu.space)
        self.intermediate = GridFunction(self.gfu.space)

        self.active_facets = BitArray(mesh.nfacet)
        self.inner_facets = BitArray(mesh.nfacet)
        self.bnd_facets = BitArray(mesh.nfacet)

        self.bnd_facets_ind = GridFunction(FacetFESpace(mesh,order=0))
        self.nobnd_facets_ind = IfPos(self.bnd_facets_ind, 0, 1)
        self.wind = wind
        if wind is not None:
            self.SetWind(wind)

    
    def SetInitialValues(self, initial_values: CoefficientFunction, initial_time: float = 0.0):
        if self.time is not None:
            self.time.Set(initial_time)
        self.gfu.Set (initial_values)
        self.ValidateStep()

    def SetWind(self, wind: CoefficientFunction):
        u, v = self.u, self.v
        fes = self.fes
        self.mass = RestrictedBilinearForm(fes, element_restriction=self.active_elements,
                                          facet_restriction=self.active_facets, check_unused=False)
        self.mass += u * v * dx(definedonelements=self.active_elements)
        self.mass.Assemble(reallocate=True)
        freedofs = GetDofsOfElements(self.fes, self.active_elements)

        self.invmass = self.mass.mat.Inverse(freedofs=freedofs)

        self.tempu = self.mass.mat.CreateColVector()

        wn = wind * specialcf.normal(self.mesh.dim)

        if not self.usetrace:
            self.bfa = RestrictedBilinearForm(fes, element_restriction=self.active_elements,
                                          facet_restriction=self.active_facets, check_unused=False)


            self.bfa += -u * wind * grad(v) * dx(definedonelements=self.active_elements, bonus_intorder=1)
            self.bfa += (wn * IfPos(wn, self.nobnd_facets_ind*u, self.nobnd_facets_ind*u.Other()) * v).Compile(self.compile, wait=True) * dx(
                element_boundary=True, definedonelements=self.active_elements)

            if self.inflow_values is not None:
                self.bfa += (wn * IfPos(wn, u, self.inflow_values) * v).Compile(self.compile, wait=True) * ds(
                skeleton=True, definedonelements=self.active_facets)
            else:
                self.bfa += (wn * u * v).Compile(self.compile, wait=True) * ds(
                skeleton=True)#, definedonelements=self.bnd_facets)


        else:
            self.fes_trace = Discontinuous(FacetFESpace(self.mesh, order=self.order))
            utr, vtr = self.fes_trace.TnT()

            self.bfa = BilinearForm(fes, nonassemble=True)
            self.bfa += -u * wind * grad(v) * dx

            self.bfa_trace = BilinearForm(self.fes_trace, nonassemble=True)
            wn = wind * specialcf.normal(self.mesh.dim)
            self.bfa_trace += ((wn * IfPos(wn, utr, utr.Other(bnd=self.inflow_values)) * vtr).Compile(
                self.compile,wait=True) * dx(element_boundary=True))



    def SetTimeStepSize(self, dt: float):
        self.dt = dt


    def Step(self):
        if self.time is not None:
            self.time.Set(self.time.Get() + self.dt)

        self.bnd_facets[:] = GetFacetsWithNeighborTypes(self.mesh, a=self.active_elements, b=~self.active_elements,
                                                        bnd_val_a=False, bnd_val_b=True)
        self.bnd_facets_ind.vec[:] = 0
        self.bnd_facets_ind.vec[self.bnd_facets] = 1

        self.active_facets[:] = GetFacetsWithNeighborTypes(self.mesh, a=self.active_elements, b=self.active_elements, use_and=False)
        self.inner_facets[:] = GetFacetsWithNeighborTypes(self.mesh, a=self.active_elements, b=self.active_elements, use_and=True)

        freedofs = GetDofsOfElements(self.fes, self.active_elements)

        if self.usetrace:
            #self.bfa_trace.Assemble(reallocate=True)
            trace = self.fes.TraceOperator(self.fes_trace, False)
            aop = self.bfa.mat + trace.T @ self.bfa_trace.mat @ trace
        else:
            self.bfa.Assemble(reallocate=True)
            aop = self.bfa.mat

        self.invMA = self.invmass @ aop

        self.tempu.data = Projector(freedofs,range=True) * self.past.vec - 0.5 * self.dt * self.invMA * self.past.vec
        self.gfu.vec.data = Projector(freedofs,range=True) * self.past.vec - self.dt * self.invMA * self.tempu


    @property
    def field(self):
        return self.gfu