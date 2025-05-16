"""
This file introduces an H1 conforming Stokes discretization for a fluid.
"""
from ngsolve import *

from .discretization import FluidDiscretization
from .params import FluidParameters, WallParameters


class H1ConformingFluid(FluidDiscretization):
    """
    This class represents Taylor-Hood elements.
    """
    def __init__(self, mesh: Mesh, fluid_params: FluidParameters, order: int = 4, levelset = None, wall_params: WallParameters = None):
        """
        Initializes an H1-conforming fluid represented by the Taylor-Hood element on our mesh.
        """
        if order < 4:
            print("WARNING: Taylor-Hood for order < 4 is not stable on all meshes.")
        super(H1ConformingFluid, self).__init__(mesh=mesh, fluid_params=fluid_params, order=order, levelset=levelset, wall_params=wall_params)

    def InitializeProblem(self, Dbndc, Dbnd, Nbndc=None, Nbnd=""):
        V = VectorH1(self.mesh, order=self.order, dirichlet=Dbnd)
        Q = H1(self.mesh, order=self.order-1)
        X = V*Q
        self.bf = BilinearForm(X)
        self.lf = LinearForm(X)
        self.fespace = X


    def SetBoundaryCondition(self, Dbndc, Dbnd, Nbndc=None, Nbnd=""):
        """
        Set the boundary conditions for your problem.

            parameters:
                Dbndc: CoefficientFunction or similar, describing the values on the Dirichlet boundary.
                Dbnd: str indicating the parts of Dirichlet boundary
                Nbndc: CoefficientFunction or similar, describing the normal derivative on the Neumann boundary, i.e. Nbndc~Dx*n.
                Nbnd: str indicating the parts of Neumann boundary
        """
        self.Dbndc = Dbndc
        self.Dbnd = Dbnd

        if Nbndc == None:
            if self.mesh.dim == 2:
                Nbndc = CF((0,0))
            elif self.mesh.dim == 3:
                Nbndc = CF((0,0,0))
            else:
                raise Exception("Other dimensions than two or three not supported")

        self.Nbndc = Nbndc
        self.Nbnd = Nbnd
        self.InitializeProblem(Dbndc=Dbndc, Dbnd=Dbnd, Nbndc=Nbndc, Nbnd=Nbnd)


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

        (u,p), (v,q) = self.fespace.TnT()
        nu = self.fluid_params["viscosity"]
        n = specialcf.normal(self.mesh.dim)

        self.bf += (nu*InnerProduct(Grad(u), Grad(v)) - div(u)*q - div(v)*p) * dx
        self.lf += rhs*v*dx + g*q*dx + nu*self.Nbndc * v * dx(definedon=self.Nbnd)


    def SolveStokes(self):
        self.bf.Assemble()
        self.lf.Assemble()

        gfu = GridFunction(self.fespace)
        gfu.vec.data[:] = 0
        gfu.components[0].Set(self.Dbndc, definedon=self.mesh.Boundaries(self.Dbnd))

        gfu.vec.data += self.bf.mat.Inverse(freedofs=self.fespace.FreeDofs()) * (self.lf.vec - self.bf.mat * gfu.vec)

        return gfu
