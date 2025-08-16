from ngsolve import *
from ngsxditto.levelset import *
from xfem import *


class DiffusionBasedVelocityExtension:
    """
    Extends a velocity field from an interface to the whole domain using a diffusion based algorithm.
    """
    def __init__(self, lset:LevelSetGeometry, gamma:float=0.1, order:int=2, ghost_stab:int=2, dirichlet:str=".*"):
        """
        Initialise the diffusion based velocity extension with the given parameters.
        Parameters:
        ----------
        lset: LevelSetGeometry
            The levelset where the velocity field is given.
        gamma: float
            The diffusion coefficient.
        order: int
            The polynomial order
        ghost_stab: int
            The ghost stabilitization coefficient.
        dirichlet: str
            The dirichlet boundary condition of the extension problem.
        """
        self.lset = lset
        self.mesh = self.lset.mesh
        self.gamma = gamma
        self.order = order
        self.ghost_stab = ghost_stab
        self.dirichlet = dirichlet
        self.V = VectorH1(self.mesh, order=self.order, dirichlet=dirichlet, dgjumps=True)

    def SolveVelocity(self, u_field: GridFunction):
        """
        Solves for the velocity field on the whole domain.
        Parameters:
        ----------
        u_field: GridFunction
            The velocity field defined on the interface.

        Returns:
        -------
        w_field: GridFunction
            The velocity field on the whole domain.
        """
        n = self.lset.n
        h = specialcf.mesh_size

        w, z = self.V.TnT()

        dx_neg = self.lset.dx_neg
        dS = self.lset.dS

        a = BilinearForm(self.V)
        a += self.gamma * h * InnerProduct((Grad(w) * n), (Grad(z) * n)) * dx_neg
        a += InnerProduct(w, n) * InnerProduct(z, n) * dS
        a += self.ghost_stab/ h * (w - w.Other()) * (z - z.Other()) * dFacetPatch(deformation=self.lset.deformation)
        a.Assemble()

        f = LinearForm(self.V)
        f += u_field * self.lset.n * InnerProduct(z, n) * dS
        f.Assemble()

        w_field = GridFunction(self.V)
        w_field.vec.data = a.mat.Inverse(self.V.FreeDofs()) * f.vec

        return w_field
