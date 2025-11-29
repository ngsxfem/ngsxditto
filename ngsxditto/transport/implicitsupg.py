from ngsolve import *
from .basetransport import BaseTransport
from ngsxditto import direct_solver_spd, direct_solver_nonspd
import typing
from xfem import *


class ImplicitSUPGTransport(BaseTransport):
    """
    This class propagates a function along a given velocity field (wind) using the Crank-Nicolson scheme
    in time and the SUPG method as space discretization.
    """
    def __init__(self, mesh, wind=None, inflow_values=None, dt=0.01, order=2, source=None,
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
        time: Parameter
            reference to a Parameter for the time (to update depending coeffiecient function during propagate)
        dt: float
            The time step size for the transport.
        """
        super().__init__(mesh, wind, inflow_values, dt, source, order=order, active_elements=active_elements)

        self.fes = H1(mesh, order=order, dgjumps=True)
        self.u, self.v = self.fes.TnT()

        self.active_facets = BitArray(mesh.nfacet)
        self.inner_facets = BitArray(mesh.nfacet)
        self.bnd_facets = BitArray(mesh.nfacet)


        self.wind = wind
        self.gamma = None

        self.bfa = RestrictedBilinearForm(self.fes, element_restriction=self.active_elements,
                                          facet_restriction=self.active_facets, check_unused=False)
        self.rhs = RestrictedBilinearForm(self.fes, element_restriction=self.active_elements,
                                          facet_restriction=self.active_facets, check_unused=False)
        self.inv = None

        self.mass_term = None
        self.conv = None

        self.gfu = GridFunction(self.fes)
        self.current = self.gfu
        self.past = GridFunction(self.gfu.space)
        self.intermediate = GridFunction(self.gfu.space)

        self.bnd_facets_ind = GridFunction(FacetFESpace(mesh,order=0))
        self.nobnd_facets_ind = IfPos(self.bnd_facets_ind, 0, 1)

        if wind is not None:
            self.SetWind(wind)




    def SetInitialValues(self, initial_values: CoefficientFunction, initial_time: float = 0.0):
        if self.time is not None:
            self.time.Set(initial_time)
        self.gfu.Set(initial_values)

    def SetWind(self, wind: CoefficientFunction):
        self.wind = wind
        self.UpdateForms()

    def UpdateForms(self):
        u, v = self.u, self.v
        h = specialcf.mesh_size
        W = L2(self.mesh, order=0)
        gamma_gfu = GridFunction(W)
        gamma_gfu.Set(h / (2 * Norm(self.wind) + 10**(-5)))
        self.gamma = CoefficientFunction(gamma_gfu)

        self.mass_term = u * (v + self.gamma * self.wind * grad(v)) * dx(definedonelements=self.active_elements)
        self.conv = self.wind * grad(u) * (v + self.gamma * self.wind * grad(v)) * dx(definedonelements=self.active_elements)

        self.bfa = BilinearForm(self.fes, symmetric=False)
        self.bfa += self.mass_term
        self.bfa += self.dt/2 * self.conv

        self.rhs = BilinearForm(self.fes)
        self.rhs += self.mass_term
        self.rhs += -self.dt / 2 * self.conv

    def SetTimeStepSize(self, dt: float):
        self.dt = dt
        self.UpdateForms()


    def Step(self):
        self.bnd_facets[:] = GetFacetsWithNeighborTypes(self.mesh, a=self.active_elements, b=~self.active_elements,
                                                        bnd_val_a=False, bnd_val_b=True)
        self.bnd_facets_ind.vec[:] = 0
        self.bnd_facets_ind.vec[self.bnd_facets] = 1
        self.active_facets[:] = GetFacetsWithNeighborTypes(self.mesh, a=self.active_elements, b=self.active_elements, use_and=False)
        self.inner_facets[:] = GetFacetsWithNeighborTypes(self.mesh, a=self.active_elements, b=self.active_elements, use_and=True)

        self.bfa.Assemble(reallocate=True)
        freedofs = GetDofsOfElements(self.fes, self.active_elements)
        self.inv = self.bfa.mat.Inverse(freedofs, inverse=direct_solver_nonspd)
        self.rhs.Assemble(reallocate=True)
        if self.time is not None:
            self.time.Set(self.time.Get() + self.dt)
        self.gfu.vec.data = self.inv @ self.rhs.mat * self.past.vec

    @property
    def field(self):
        return self.gfu