from ngsolve import *
from .basetransport import BaseTransport
from ngsxditto import direct_solver_spd, direct_solver_nonspd
from xfem import *
import typing

# taken and adapted from NGSolve's modeltemplates

class ImplicitDGTransport(BaseTransport):
    """
    This class propagates a function along a given velocity field (wind) using an implicit Euler discretization
    in time and the DG method as space discretization. ### TODO: Trefftz or HDG?
    """
    def __init__(self, mesh, wind=None, inflow_values=None, dt=0.01, order: int=2, source=None,
                 active_elements: typing.Optional[BitArray] = None):
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
        active_elements: BitArray
            submesh defined by BitArray on which the transport is defined. If None, the whole mesh is used.
        """
        super().__init__(mesh, wind, inflow_values, dt, source, active_elements=active_elements, order=order)

        self.fes = L2(mesh, order=order, all_dofs_together=True, dgjumps=True)

        self.active_facets = BitArray(mesh.nfacet)
        self.bnd_facets = BitArray(mesh.nfacet)

        if active_elements is None:
            self.active_elements = BitArray(mesh.ne)
            self.active_elements[:] = True
        else:
            self.active_elements = active_elements


        self.gfu = GridFunction(self.fes)
        self.current = self.gfu
        self.past = GridFunction(self.gfu.space)
        self.intermediate = GridFunction(self.gfu.space)
        self.past_cont = GridFunction(H1(mesh, order=order, dgjumps=True))

        self.bnd_facets_ind = GridFunction(FacetFESpace(mesh,order=0))
        self.nobnd_facets_ind = IfPos(self.bnd_facets_ind, 0, 1)

        self.inflow_values = inflow_values if inflow_values else self.past_cont   # if no inflow values are given, we use the past solution

        if wind is not None:
            self.SetWind(wind)


    def SetInitialValues(self, initial_values: CoefficientFunction, initial_time: float = 0.0):
        if self.time is not None:
            self.time.Set(initial_time)
        self.gfu.Set (initial_values) #, definedonelements=self.active_elements)
        self.past_cont.Set (initial_values)
        self.ValidateStep()

    def SetWind(self, wind: CoefficientFunction):
        u, v = self.gfu.space.TnT()
        n = specialcf.normal(self.mesh.dim)

        self.bfa = RestrictedBilinearForm(self.fes, element_restriction=self.active_elements,
                                          facet_restriction=self.active_facets, check_unused=False)
        self.bfa += u * v * dx(definedonelements=self.active_elements)
        self.bfa += self.dt*(v * (wind | grad(u))).Compile() * dx(definedonelements=self.active_elements, bonus_intorder=1) # ! bonus_intorder=1 important, because integration order would be too low otherwise
        self.bfa += self.dt*(- IfPos((wind|n), 0, (wind|n) * (u - self.nobnd_facets_ind *u.Other())) * v).Compile() * dx(element_boundary=True, definedonelements=self.active_elements)

        self.lf = LinearForm(self.fes)
        self.lf += ( self.past_cont * v).Compile() * dx(definedonelements=self.active_elements)
        self.lf += -self.dt*(IfPos((wind|n), 0, (wind|n) * self.inflow_values * v * self.bnd_facets_ind)).Compile() * dx(definedonelements=self.active_elements, element_boundary=True, bonus_intorder=1)  # integral on boundary facets

        ### TODO: test agains dx(skeleton=True)

    def SetTimeStepSize(self, dt: float):
        self.dt = dt

    def Step(self):
        self.active_facets[:] = GetFacetsWithNeighborTypes(self.mesh, a=self.active_elements, b=self.active_elements, use_and=False) ##todo
        self.bnd_facets[:] = GetFacetsWithNeighborTypes(self.mesh, a=self.active_elements, b=~self.active_elements,
                                                        bnd_val_a=False, bnd_val_b=True)
        self.bnd_facets_ind.vec[:] = 0
        self.bnd_facets_ind.vec[self.bnd_facets] = 1

        with TaskManager():
            self.bfa.Assemble(reallocate=True)
            self.lf.Assemble()

        if self.time is not None:
            self.time.Set(self.time.Get() + self.dt)

        freedofs = GetDofsOfElements(self.fes, self.active_elements)
        with TaskManager():
            self.gfu.vec.data = self.bfa.mat.Inverse(freedofs = freedofs, inverse = direct_solver_nonspd) * self.lf.vec

    @property
    def field(self):
        return self.gfu