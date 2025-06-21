"""
This file introduces an H1 conforming Stokes discretization for a fluid.
"""
from ngsolve import *

from .discretization import FluidDiscretization
from .params import FluidParameters, WallParameters


class TaylorHood(FluidDiscretization):
    """
    This class represents Taylor-Hood elements.
    """
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, levelset = None, wall_params: WallParameters = None):
        """
        Initializes an H1-conforming fluid represented by the Taylor-Hood element on our mesh.
        """
        if order < 4:
            print("WARNING: Taylor-Hood for order < 4 is not stable on all meshes.")
        super().__init__(mesh=mesh, fluid_params=fluid_params, order=order, levelset=levelset, wall_params=wall_params)
        self.nu = self.fluid_params["viscosity"]


    def InitializeProblem(self, dbnd):
        V = VectorH1(self.mesh, order=self.order, dirichlet=dbnd)
        Q = H1(self.mesh, order=self.order - 1)
        X = V * Q
        (self.u, self.p), (self.v, self.q) = X.TnT()
        (u, p), (v, q) = (self.u, self.p), (self.v, self.q)

        self.mass = u * v * dx

        self.stokes = (self.nu * InnerProduct(grad(u), grad(v)) +
                       div(u) * q + div(v) * p - 1e-10 * p * q) * dx

        self.a = BilinearForm(X)
        self.a += self.stokes
        self.lf = LinearForm(X)
        self.gfu = GridFunction(X)
        self.fes = X



    def SetBoundaryConditions(self, dirichlet:dict=None, neumann:dict=None):
        """
        Set the non-zero dirichlet and neumann boundary conditions for your problem.

            parameters:
                Dbndc: CoefficientFunction or similar, describing the values on the Dirichlet boundary.
                Dbnd: str indicating the parts of Dirichlet boundary
                Nbndc: CoefficientFunction or similar, describing the normal derivative on the Neumann boundary, i.e. Nbndc~Dx*n.
                Nbnd: str indicating the parts of Neumann boundary
        """
        if dirichlet is None:
            dirichlet = {}

        if neumann is None:
            neumann = {}

        self.dirichlet = dirichlet
        self.neumann = neumann


    def InitializeVarForm(self, rhs: CoefficientFunction = None):
        """
        Initialize the variational formulation.
        Currently only homogeneous Dirichlet boundary conditions.

            parameters:
                rhs: The right hand side f of your variational formulation.
        """
        if rhs == None:
            rhs = CF((0,0)) if self.mesh.dim == 2 else CF((0,0,0))

        g = CF(0) # divergence constraint: I think we never want nonzero, due to mass conservation of our fluids?

        (u,p), (v,q) = (self.u, self.p), (self.v, self.q)
        n = specialcf.normal(self.mesh.dim)

        self.a += self.stokes
        self.a.Assemble()
        self.lf += rhs*v*dx + g*q*dx
        for (region, fct) in self.neumann.items():
            self.lf += self.nu*fct * v * dx(definedon=region)

        self.lf.Assemble()
        self.conv = BilinearForm(self.fes, nonassemble=True)
        self.conv += (Grad(u) * u) * v * dx

        self.m_star = BilinearForm(self.fes)
        self.m_star += self.mass + self.dt * self.stokes
        self.m_star.Assemble()

        self.inv = self.m_star.mat.Inverse(freedofs=self.fes.FreeDofs(), inverse="sparsecholesky")


    def SolveStokes(self):
        gfu = GridFunction(self.fes)
        for (region, fct) in self.dirichlet.items():
            gfu.components[0].Set(fct, definedon=self.mesh.Boundaries(region))

        gfu.vec.data += self.a.mat.Inverse(freedofs=self.fes.FreeDofs()) * (self.lf.vec - self.a.mat * gfu.vec)
        return gfu

    def OneStep(self):
        res = self.conv.Apply(self.gfu.vec) + self.a.mat*self.gfu.vec
        self.gfu.vec.data -= self.dt * self.inv * res

