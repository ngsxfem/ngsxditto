from ngsolve import *
from xfem import *
from xfem.lsetcurv import *
from typing import Optional, Tuple, Union
from ngsxditto.stepper import *
import ngsolve.webgui as ngw


class MeanCurvatureSolver(StatelessStepper):
    """
    Class to compute the mean curvature vector from a level set function
    """

    def __init__(self, mesh: Mesh, order: int = 1,
                 lset=None, own_cutinfo=False, own_lsetadap=False,
                 gp_param: Union[None, float, CoefficientFunction] = specialcf.mesh_size):
        """
        Initialize the mean curvature solver with a mesh discretization parameters, 
        and (if existing anyway) lsetadap and cutinfo

        Parameters
        ----------
        mesh : Mesh
            The computational mesh
        order : int
            The polynomial order of the sought for mean curvature vector
        lset: LevelSetGeometry
            The levelset we want to know the mean curvature of
        own_cutinfo: bool
            If a new CutInfo object should be created
        own_lsetadap: bool
            If a new LSetAdap object should be created
        gp_param : float or CoefficientFunction or None
            The parameter for the generalized Poisson problem
        """
        super().__init__()

        self.mesh = mesh
        self.order = order
        self.lset = lset
        self.gp_param = gp_param

        # Level-Set Adaptation
        self.own_lsetadap = own_lsetadap

        if own_lsetadap:
            self.own_lsetadap = True
            self.lsetadap = LevelSetMeshAdaptation (mesh, order=order, threshold=0.5,
                                                    discontinuous_qn=True)
        else:
            if lset is not None:
                self.lsetadap = self.lset.lsetadap
            else:
                self.lsetadap = None

        if lset is not None:
            self.lset_approx = self.lsetadap.lset_p1
        else:
            self.lset_approx = None
        
        # cut info and XFEM context
        self.own_cutinfo = own_cutinfo

        if own_cutinfo:
            self.cutinfo = CutInfo(mesh)
        else:
            if self.lset is not None:
                self.cutinfo = self.lset.cutinfo
            else:
                self.cutinfo = None

        self.X = VectorH1(mesh, order=order, dgjumps=(gp_param!=None))

        self.H = GridFunction(self.X)


    def SetLset(self, lset):
        self.lset = lset

        self.cutinfo = self.lset.cutinfo
        self.lsetadap = self.lset.lsetadap
        self.lset_approx = self.lsetadap.lset_p1
        self.own_cutinfo = False
        self.own_lsetadap = False


    def Step(self):
        """
        Solve for and update the mean curvature vector.
        """
        if self.own_lsetadap:
            self.lsetadap.CalcDeformation(self.lset.field)
        if self.own_cutinfo:
            self.cutinfo.Update(self.lsetadap.lset_p1)

        u, v = self.X.TnT()
        h = specialcf.mesh_size

        # normal vector from level set
        grad_phi = grad(self.lsetadap.lset_p1)
        n = Normalize(grad_phi)
        # norm_grad_phi = sqrt(InnerProduct(grad_phi, grad_phi)) + 1e-10
        # n = grad_phi / norm_grad_phi

        ifels = self.cutinfo.GetElementsOfType(IF)
        ds = self.ds = dCut(self.lsetadap.lset_p1, IF, definedonelements=ifels, deformation=self.lsetadap.deform)
        dX = dx(definedonelements=ifels, deformation=self.lsetadap.deform)

        # tangent projector
        E = Id(self.mesh.dim)
        P = E - OuterProduct(n, n)

        facets = GetFacetsWithNeighborTypes(self.mesh, a=ifels, b=ifels)
        if self.gp_param is None:
            facets[:] = False
        else:
            dw = dFacetPatch(definedonelements=facets, deformation=self.lsetadap.deform)
        # bilinear form
        a = RestrictedBilinearForm(self.X, element_restriction=ifels, 
                                   facet_restriction=facets, check_unused=False)
        a += u*v * ds + h * (grad(u) * n) * (grad(v) * n) * dX
        if self.gp_param is not None:
            a += self.gp_param * (u-u.Other()) * (v-v.Other()) * dw
        a.Assemble()

        # linear form
        f = LinearForm(self.X)
        f += InnerProduct(P*E, P*grad(v)) * ds
        f.Assemble()

        # solution
        self.H.vec[:] = 0.0
        #self.freedofs = self.X.FreeDofs()
        self.freedofs = GetDofsOfElements(self.X, self.cutinfo.GetElementsOfType(IF))

        self.H.vec.data = a.mat.Inverse(self.freedofs) * f.vec


    def compute_l2_error(self, H):
        return sqrt(Integrate(InnerProduct(H-self.H,H-self.H) * self.ds, mesh=self.mesh))


if __name__ == "__main__":
    from netgen.geom2d import SplineGeometry
    from ngsolve import sqrt, x, y, Mesh, specialcf

    geo = SplineGeometry()
    geo.AddRectangle([-2, -2], [2, 2], bc=1)
    mesh = Mesh(geo.GenerateMesh(maxh=0.1))

    phi = sqrt(x*x + y*y) - 1
    h = specialcf.mesh_size
    solver = MeanCurvatureSolver(mesh, order=1, gp_param=0.5*h*h)
    solver.compute(phi)
    Hexact = CF((x, y))
    err = solver.compute_l2_error(Hexact)
    print("L2-error: ", err)
    from netgen import gui
    Draw(BitArrayCF(solver.cutinfo.GetElementsOfType(IF))*solver.H, mesh, "H")    
    Draw(BitArrayCF(solver.cutinfo.GetElementsOfType(IF))*solver.H-BitArrayCF(solver.cutinfo.GetElementsOfType(IF))*Hexact, mesh, "Herr")    
    Draw(BitArrayCF(solver.cutinfo.GetElementsOfType(IF))*Hexact, mesh, "Hexact")