from xfem import *
import ngsolve.webgui as ngw
from ngsolve import *
import numpy as np

class BaseGradientTester:
    def __init__(self, mesh):
        self.mesh = mesh

    def MinMaxGradientNorm(self, phi):
        raise NotImplementedError("MinMaxGradientNorm not implemented")


class NaiveGradientTester(BaseGradientTester):
    def __init__(self, mesh):
        super().__init__(mesh)

    def MinMaxGradientNorm(self, phi):
        norm_grad = Norm(grad(phi))
        V = H1(self.mesh, order=1)
        gfu = GridFunction(V)
        gfu.Set(norm_grad)

        max_grad = max(gfu.vec.data)
        min_grad = min(gfu.vec.data)

        return min_grad, max_grad


class ElementBand(BaseGradientTester):
    def __init__(self, mesh):
        super().__init__(mesh)

    def MinMaxGradientNorm(self, phi, iterations=1):
        P1 = H1(self.mesh, order=1)
        phi_p1 = GridFunction(P1)
        phi_p1.Set(phi)
        #V = phi.space
        ci = CutInfo(self.mesh, phi_p1)
        levelset_band = ci.GetElementsOfType(IF)
        for _ in range(2*iterations):
            neighbouring_facets = GetFacetsWithNeighborTypes(self.mesh,a=levelset_band,b=levelset_band,
                                           bnd_val_a=False,bnd_val_b=False,use_and=False)
            levelset_band = GetElementsWithNeighborFacets(self.mesh,neighbouring_facets)

        active_dofs = GetDofsOfElements(P1, levelset_band)

        norm_grad = Norm(grad(phi))
        gfu = GridFunction(P1)
        gfu.Set(norm_grad)

        if active_dofs:
            max_grad = np.max(np.array(gfu.vec.data)[active_dofs])
            min_grad = np.min(np.array(gfu.vec.data)[active_dofs])
        else:
            min_grad, max_grad = None, None
        print(min_grad, max_grad)
        return min_grad, max_grad


