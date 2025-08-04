from ngsolve import *
from ngsxditto.levelset import *
from xfem import *


class VelocityExtension:
    def __init__(self, lset, gamma=0.1, order=2, ghost_stab=2, dirichlet=".*"):
        self.lset = lset
        self.mesh = self.lset.mesh
        self.gamma = gamma
        self.order = order
        self.ghost_stab = ghost_stab
        self.dirichlet = dirichlet
        self.V = VectorH1(self.mesh, order=self.order, dirichlet=dirichlet, dgjumps=True)

    def SolveVelocity(self, u_field):
        n = self.lset.n
        h = specialcf.mesh_size

        w, z = self.V.TnT()

        a = BilinearForm(self.V)
        a += self.gamma * h * InnerProduct((Grad(w) * n), (Grad(z) * n)) * self.lset.dx_neg
        a += InnerProduct(w, n) * InnerProduct(z, n) * self.lset.dS
        a += self.ghost_stab/ h * (w - w.Other()) * (z - z.Other()) * dFacetPatch(deformation=self.lset.deformation)
        a.Assemble()

        f = LinearForm(self.V)
        f += u_field * self.lset.n * InnerProduct(z, n) * self.lset.dS
        f.Assemble()

        w_field = GridFunction(self.V)
        w_field.vec.data = a.mat.Inverse(self.V.FreeDofs()) * f.vec

        return w_field
