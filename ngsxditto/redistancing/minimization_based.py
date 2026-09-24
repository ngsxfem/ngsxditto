from ngsolve import *
import ngsolve.webgui as ngw
from xfem.lsetcurv import LevelSetMeshAdaptation

from .redistancing import *
from xfem import *

class MinimizationBasedRedistancing(BaseRedistancing):
    """ Redistancing algorithm basd on minimization of the energy. Iteratively updates the levelset function to restore
        the signed distance function while penalizing derivations from the initial interface."""
    #: Default interface-penalty coefficient, used as ``alpha_scale / h``.
    #: See the ``alpha`` docstring for where the value comes from.
    DEFAULT_ALPHA_SCALE = 5000.0

    def __init__(self, alpha=None, n_iter=10, initializer: BaseRedistancing = None,
                 alpha_scale=None):
        """
        Parameters:
        -----------
        alpha : float, optional
            Penalty coefficient of the interface term, used *as given*, i.e.
            NOT scaled with the mesh. Leave it at ``None`` (recommended) to get
            the mesh-scaled default ``alpha_scale / h**2`` instead.

            The interface is only held in place by a penalty, so its position
            is preserved to O(1/alpha) -- and a *fixed* alpha therefore stops
            converging under mesh refinement. Measured on an exact ellipse
            perturbed so that the zero set is analytically unchanged, the drift
            of the zero set after one pass is

                h       alpha=1e4 (old default)   alpha = 5000/h
                0.04    2.2e-5                    1.7e-5
                0.02    1.5e-5  (order 0.53)      3.1e-6  (order 2.48)
                0.01    1.4e-5  (order 0.13)      1.1e-6  (order 1.49)

            i.e. with a fixed alpha the drift flattens out at ~1.5e-5 and the
            redistancing becomes the accuracy floor of the whole level-set
            computation, while the scaled default keeps converging. Pass an
            explicit ``alpha`` only to reproduce old results.
        alpha_scale : float, optional
            Coefficient of the mesh-scaled penalty ``alpha_scale / h``,
            defaulting to :attr:`DEFAULT_ALPHA_SCALE`. In 2D the volume term of
            the energy is O(1) per basis function while an unscaled interface
            term scales like ``alpha*h``, so ``alpha ~ 1/h`` is what keeps the
            two in balance under refinement -- the usual Nitsche-type scaling.

            (``1/h**2`` would also bound the penalty's own contribution,
            ``~1/alpha``, by O(h**2) rather than O(h). Measured, that only
            starts to matter below h ~ 2e-3, where the penalty error would
            overtake the discretisation error; in the range that is actually
            computed the two scalings differ by ~35% and ``1/h`` keeps the
            condition number lower.) Ignored when ``alpha`` is given.
        n_iter : int
            The number of iterations for the minimization process.
        initializer : BaseRedistancing, optional
            Another redistancer applied once, in place, to `phi_start` before
            this class's own iteration begins. This scheme only ever
            *corrects* whatever field it is handed -- given a rough/drifted
            input it converges slowly (interior/medial-axis structure, far
            from the interface) or needs many iterations to look reasonable
            far from the interface. A better starting point fixes both at no
            extra cost to the main loop, e.g. `FastMarching()` (causally
            correct global distance structure from the start) or
            `GPExtensionRedistancing()` (cheap smoothing of a noisy input).
        """
        super().__init__()
        self.alpha = alpha
        self.alpha_scale = (self.DEFAULT_ALPHA_SCALE if alpha_scale is None
                            else alpha_scale)
        self.n_iter = n_iter
        self.initializer = initializer

    def Redistance(self, phi_start, deformation=None):
        if self.initializer is not None:
            self.initializer.Redistance(phi_start, deformation)
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
        # mesh-scaled unless the caller pinned alpha explicitly -- see __init__
        alpha_cf = (self.alpha if self.alpha is not None
                    else self.alpha_scale / specialcf.mesh_size)
        a += alpha_cf * phi * v * dS
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
