from ngsolve import *
from ngsxditto.levelset import *
from ngsxditto.stepper import *
from ngsxditto import direct_solver_spd, direct_solver_nonspd
from xfem import *
import ngsolve.webgui as ngw

class LevelsetBasedExtension(StatelessStepper):
    """
    Extends a vector field from an interface to the whole domain using a diffusion based algorithm.
    """
    def __init__(self, lset:LevelSetGeometry, rhs=None, gamma:float=0.1, order:int=2, ghost_stab:int=2, dirichlet:str=".*",
                 q: CoefficientFunction=CF(0)):
        """
        Initialise the diffusion based vector extension with the given parameters.

        Parameters:
        -----------
        lset: LevelSetGeometry
            The levelset where the vector field is given.
        gamma: float
            The diffusion coefficient.
        order: int
            The polynomial order
        ghost_stab: int
            The ghost stabilitization coefficient.
        dirichlet: str
            The dirichlet boundary condition of the extension problem.
        """
        super().__init__()
        self.lset = lset
        self.mesh = self.lset.mesh
        self.gamma = gamma
        self.order = order
        self.ghost_stab = ghost_stab
        self.dirichlet = dirichlet
        self.V = VectorH1(self.mesh, order=self.order, dirichlet=dirichlet, dgjumps=True)
        self.field = GridFunction(self.V)
        self.current = self.field
        self.rhs = rhs
        self.q = q


    def SetRhs(self, rhs):
        self.rhs = rhs

    def Step(self):
        """
        Solves for the vector field on the whole domain.

        Parameters:
        -----------
        u_field: GridFunction
            The vector field defined on the interface.
        """
        n = self.lset.n
        h = specialcf.mesh_size

        w, z = self.V.TnT()

        dx_neg = self.lset.dx_neg
        dS = self.lset.dS

        a = BilinearForm(self.V)
        a += self.gamma * h * InnerProduct((Grad(w) * n), (Grad(z) * n)) * dx_neg
        a += InnerProduct(w, n) * InnerProduct(z, n) * dS
        a += self.ghost_stab/h * (w - w.Other()) * (z - z.Other()) * dFacetPatch(deformation=self.lset.deformation)
        a.Assemble()

        f = LinearForm(self.V)
        f += (self.rhs * self.lset.n + self.q) * InnerProduct(z, n) * dS
        f.Assemble()

        deformed_lsetp1_field = GridFunction(self.V)
        deformed_lsetp1_field.vec.data = a.mat.Inverse(self.V.FreeDofs(), inverse=direct_solver_spd) * f.vec

        self.field.Set(shifted_eval(deformed_lsetp1_field, back=self.lset.deformation, forth=None))


