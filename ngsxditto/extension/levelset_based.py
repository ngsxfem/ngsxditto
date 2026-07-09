from ngsolve import *
from ngsxditto.levelset import *
from ngsxditto.stepper import *
from ngsxditto.extrapolation import ExtrapolatorSource
from ngsxditto import direct_solver_spd, direct_solver_nonspd
from xfem import *
import ngsolve.webgui as ngw

class LevelsetBasedExtension(ExtrapolatorSource, StatelessStepper):
    """
    Extends a vector field from an interface to the whole domain using a diffusion based algorithm.
    """
    def __init__(self, lset:LevelSetGeometry, rhs=None, gamma:float=0.1, order:int=2, ghost_stab:int=1,
                 no_slip:str= ".*", no_penetration:str= "", q: CoefficientFunction=CF(0)):
        """
        Initialise the diffusion based vector extension with the given parameters.

        Parameters:
        -----------
        lset: LevelSetGeometry
            The levelset where the vector field is given.
        rhs: CoefficientFunction|None
            The vector-valued function that should be extended.
        gamma: float
            The diffusion coefficient.
        order: int
            The polynomial order
        ghost_stab: int
            The ghost stabilitization coefficient.
        no_slip: str
            The boundary where the vector field should be zero.
        no_penetration: str
            The boundary where the normal component of the vector field should be zero.
        q: CoefficientFunction
            A scalar function that is added to the rhs * normal term.
        """
        super().__init__()
        self.lset = lset
        self.mesh = self.lset.mesh
        self.gamma = gamma
        self.order = order
        self.ghost_stab = ghost_stab
        self.no_slip = no_slip
        self.no_penetration = no_penetration
        self.V = VectorH1(self.mesh, order=self.order, dirichlet=no_slip, dgjumps=True)
        components = list(self.V.components)

        self.free_dofs = self.V.FreeDofs()

        normal_dofs = BitArray(self.V.ndof)
        normal_dofs[:] = False

        if self.mesh.dim == 2:
            self.V_x, self.V_y = components
            offset_y = self.V_x.ndof

            bnd_x = self.V_x.GetDofs(self.mesh.Boundaries("left|right"))
            bnd_y = self.V_y.GetDofs(self.mesh.Boundaries("top|bottom"))

            for i, is_on_bnd in enumerate(bnd_x):
                if is_on_bnd:
                    normal_dofs[i] = True
            for i, is_on_bnd in enumerate(bnd_y):
                if is_on_bnd:
                    normal_dofs[offset_y + i] = True
        else:
            self.V_x, self.V_y, self.V_z = components
            offset_y = self.V_x.ndof
            offset_z = self.V_x.ndof + self.V_y.ndof

            _bnd_names = self.mesh.GetBoundaries()
            x_bnds = "|".join(b for b in _bnd_names if b in ("left", "right"))
            y_bnds = "|".join(b for b in _bnd_names if b in ("top", "bottom"))
            z_bnds = "|".join(b for b in _bnd_names if b in ("front", "back"))

            if x_bnds:
                for i, v in enumerate(self.V_x.GetDofs(self.mesh.Boundaries(x_bnds))):
                    if v: normal_dofs[i] = True
            if y_bnds:
                for i, v in enumerate(self.V_y.GetDofs(self.mesh.Boundaries(y_bnds))):
                    if v: normal_dofs[offset_y + i] = True
            if z_bnds:
                for i, v in enumerate(self.V_z.GetDofs(self.mesh.Boundaries(z_bnds))):
                    if v: normal_dofs[offset_z + i] = True

        zero_normal_dofs = normal_dofs & self.V.GetDofs(self.mesh.Boundaries(self.no_penetration))
        self.free_dofs &= ~zero_normal_dofs

        self.field = GridFunction(self.V)
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
        deformed_lsetp1_field.vec.data = a.mat.Inverse(self.free_dofs, inverse=direct_solver_spd) * f.vec

        self.field.Set(shifted_eval(deformed_lsetp1_field, back=self.lset.deformation, forth=None))
