from xfem import *
import ngsolve.webgui as ngw
from ngsolve import *
import numpy as np

class BaseGradientTester:
    """
    A base class for methods to determine minimal and maximal gradients in given subsets of the domain.
    """
    def __init__(self, mesh: Mesh):
        """
        Initializes the gradient tester.

        Parameters:
        -----------
        mesh: Mesh
            The computational mesh.
        """
        self.mesh = mesh

    def MinMaxGradientNorm(self, phi: CoefficientFunction):
        """
        Calculate the minimal and maximal gradient norm.

        Parameters:
        -----------
        phi: CoefficientFunction
            The function we want to know the gradient norms of.

        Returns:
        --------
        tuple[float, float]
            The minimal and maximal gradient norm.
        """
        raise NotImplementedError("MinMaxGradientNorm not implemented")


class NaiveGradientTester(BaseGradientTester):
    def __init__(self, mesh: Mesh):
        super().__init__(mesh)

    def MinMaxGradientNorm(self, phi: CoefficientFunction):
        norm_grad = Norm(grad(phi))
        V = H1(self.mesh, order=1)
        gfu = GridFunction(V)
        gfu.Set(norm_grad)

        max_grad = max(gfu.vec.data)
        min_grad = min(gfu.vec.data)

        return min_grad, max_grad


class ElementBand(BaseGradientTester):
    """
    Tests the gradient in an element band around the levelset.
    """
    def __init__(self, mesh: Mesh):
        super().__init__(mesh)

    def MinMaxGradientNorm(self, phi: CoefficientFunction, iterations: int = 1):
        """
        Calculate the minimal and maximal gradient norm in the band.

        Parameters:
        -----------
        phi: CoefficientFunction
            The function we want to know the gradient norms of.
        iterations: int
            The thickness of the element band.
        """
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
        return min_grad, max_grad


class FixedDistanceBand(BaseGradientTester):
    """
    Tests the gradient in a band around the levelset with a fixed radius.
    """
    def __init__(self, mesh):
        super().__init__(mesh)

    def MinMaxGradientNorm(self, phi, bandwidth=None):
        P1 = H1(self.mesh, order=1)
        phi_p1 = GridFunction(P1)
        phi_p1.Set(phi)

        ci = CutInfo(self.mesh, phi_p1)
        levelset_elements = ci.GetElementsOfType(IF)
        active_dofs = GetDofsOfElements(P1, levelset_elements)

        raise NotImplementedError("MinMaxGradientNorm not yet implemented")



