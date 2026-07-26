from ngsolve import *
from xfem import *

from .redistancing import *


class GPExtensionRedistancing(BaseRedistancing):
    """A single ghost-penalty diffusion/extension solve -- the scalar
    counterpart of the vector-field extension used elsewhere
    (`ngsxditto.extension.LevelsetBasedExtension`), pinned to zero on the
    interface instead of matching a normal-velocity boundary condition.

    This is **not really a redistancer**: it does not iterate toward
    |grad(phi)|=1 and should not be used as the final result of a
    redistancing step. What it does well is produce, in one cheap linear
    solve, a smooth field that already respects the interface -- which makes
    it a good (and much cheaper) alternative to `FastMarching` as the
    `initializer` for `MinimizationBasedRedistancing`: instead of handing
    that class's Newton iteration a rough/noisy input to slowly smooth out
    over many iterations, hand it something already smooth.

    One linear solve (dS = interface, dw = ghost-penalty facet coupling,
    mass-anchored to the input so it does not just decay to zero):

        (eps*phi*v + alpha*phi*v*dS + gp_stab/h*(phi-phi.Other())*(v-v.Other())*dw) . phi_new
            = eps * phi_start * v
    """

    def __init__(self, alpha=10000, gp_stab=100.0, eps=1.0):
        """
        Parameters:
        -----------
        alpha : float
            Penalty parameter pinning phi=0 on the (P1-interpolated) interface.
        gp_stab : float
            Ghost-penalty (facet-jump) diffusion strength.
        eps : float
            Mass-matrix anchoring weight against the input field; keeps the
            far field close to `phi_start` rather than drifting to zero.
        """
        super().__init__()
        self.alpha = alpha
        self.gp_stab = gp_stab
        self.eps = eps

    def Redistance(self, phi_start, deformation=None):
        mesh = phi_start.space.mesh
        order = phi_start.space.globalorder
        lsetp1 = GridFunction(H1(mesh, order=1))
        InterpolateToP1(phi_start, lsetp1)
        ci = CutInfo(mesh, lsetp1)
        anyels = ci.GetElementsOfType(ANY)
        hasif = ci.GetElementsOfType(IF)
        dS = dCut(levelset=lsetp1, domain_type=IF, definedonelements=hasif, deformation=deformation)
        # dgjumps space so the facet-patch ghost penalty (.Other() across a patch) is available
        fes = H1(mesh, order=order, dgjumps=True)
        phi, v = fes.TnT()
        dX = dx(deformation=deformation, definedonelements=anyels)
        dw = dFacetPatch(deformation=deformation)
        h = specialcf.mesh_size

        current_phi = GridFunction(fes)
        current_phi.Set(phi_start)     # project from the (possibly non-dgjumps) input space

        a = BilinearForm(fes, check_unused=False)
        a += self.eps * phi * v * dX
        a += self.alpha * phi * v * dS
        a += self.gp_stab / h * (phi - phi.Other()) * (v - v.Other()) * dw
        a.Assemble()

        b = LinearForm(fes)
        b += self.eps * current_phi * v * dX
        b.Assemble()

        freedofs = GetDofsOfElements(fes, anyels)
        current_phi.vec.data = a.mat.Inverse(freedofs) * b.vec

        phi_start.Set(current_phi)
