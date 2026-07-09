from ngsolve import *
import ngsolve.webgui as ngw
from xfem.lsetcurv import LevelSetMeshAdaptation

from .redistancing import *
from xfem import *

class MinimizationBasedRedistancing(BaseRedistancing):
    """ Redistancing algorithm basd on minimization of the energy. Iteratively updates the levelset function to restore
        the signed distance function while penalizing derivations from the initial interface."""
    def __init__(self, alpha=10000, n_iter=10):
        """
        Parameters:
        -----------
        alpha : float
            The penalty parameter for the energy functional. Higher values enforce stronger adherence to the initial interface.
        n_iter : int
            The number of iterations for the minimization process.
        """
        super().__init__()
        self.alpha = alpha
        self.n_iter = n_iter

    def Redistance(self, phi_start, deformation=None):
        mesh = phi_start.space.mesh
        order = phi_start.space.globalorder
        lsetp1 = GridFunction(H1(mesh, order=1))
        InterpolateToP1(phi_start, lsetp1)
        ci = CutInfo(mesh, lsetp1)
        hasif = ci.GetElementsOfType(IF)
        dS = dCut(levelset=lsetp1, domain_type=IF, definedonelements=hasif, deformation=deformation)
        fes = H1(mesh, order=order)
        phi, v = fes.TnT()
        dX = dx(deformation=deformation, definedonelements=ci.GetElementsOfType(ANY))
        dX_away = dx(deformation=deformation, definedonelements=ci.GetElementsOfType(UNCUT))
        a = BilinearForm(fes, check_unused=False)
        a += grad(phi) * grad(v) * dX
        a += self.alpha * phi * v * dS
        a.Assemble()

        freedofs = GetDofsOfElements(fes, ci.GetElementsOfType(ANY))
        inv = a.mat.Inverse(freedofs)
        current_phi = GridFunction(fes)
        current_phi.vec.data = phi_start.vec

        for i in range(self.n_iter):
            norm_grad = Norm(grad(current_phi))
            #d_3_grad_phi = IfPos(norm_grad - CF(1), CF(1) - 1/norm_grad, CF(2*norm_grad**2 - 3*norm_grad + 1))
            d_1_grad_phi = CF(1) - 1/norm_grad
            b = LinearForm(fes)
            b += -(d_1_grad_phi - 1) * grad(current_phi) * grad(v) * dX
            b.Assemble()

            current_phi.vec.data = inv * b.vec

        phi_start.vec.data = current_phi.vec

        undeformed_gfu = GridFunction(fes)
        undeformed_gfu.Set(shifted_eval(phi_start, back=self.deformation, forth=None))
        phi_start.vec.data = undeformed_gfu.vec
