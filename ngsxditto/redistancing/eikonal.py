from ngsolve import *
from ngsolve.nonlinearsolvers import NewtonSolver
from .redistancing import *
from xfem import dCut, IF

class EikonalRedistancing(BaseRedistancing):
    """
    This class handles redistancing approximating the Eikonal equation. 

    Idea from: "Iterative method for solving the eikonal equation", Conference Paper in Proceedings of SPIE - The International Society for Optical Engineering · November 2016

    First solve an elliptic version of the Eikonal equation and then apply an iteration over linear elliptic PDEs to improve the solution.

    """
    def __init__(self, gfphi: GridFunction=None, bandwidth: float=None, deformation: GridFunction=None):
        super().__init__(bandwidth)
        if bandwidth is not None:
            raise NotImplementedError("EikonalRedistancing does not support bandwidth so far")
        self.gfphi = gfphi
        self.deformation = deformation
        fes = gfphi.space

        ds = dCut(gfphi, IF, deformation=deformation)

        phi, psi = fes.TnT()

        h = specialcf.mesh_size

        ah = BilinearForm(fes)
        ah += 1e6/h * phi * psi * ds 
        ah += h * grad(phi) * grad(psi) * dx + (grad(phi) * grad(phi) - 1) * psi * dx

        self.solver = NewtonSolver(ah, gfphi)
        print("WARNING: EikonalRedistancing is very experimental - not meant for use in 'production' yet.")

    def Redistance(self, phi: GridFunction):
        print("WARNING: EikonalRedistancing is very experimental - not meant for use in 'production' yet.")
        if self.deformation is not None:
            self.gfphi.space.mesh.SetDeformation(self.deformation)
        self.solver.Solve(maxit=500, printing=True, callback=None, dampfactor=0.01)
        if self.deformation is not None:
            self.gfphi.space.mesh.UnsetDeformation()

