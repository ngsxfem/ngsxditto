from ngsolve import *
from xfem import *
from xfem.lsetcurv import *
from typing import Optional, Tuple, Union

class MeanCurvatureSolver:
    """
    Class to compute the mean curvature vector from a level set function
    """

    def __init__(self, mesh: Mesh, order: int = 1, 
                 cutinfo: Optional[CutInfo] = None,
                 lsetadap: Optional[LevelSetMeshAdaptation] = None,
                 gp_param: Union[None, float, CoefficientFunction] = None):
        """
        Initialisiere den Solver mit Mesh, Level-Set-Funktion und Diskretisierungsparametern.
        """

        self.mesh = mesh
        self.order = order
        self.gp_param = gp_param

        # Level-Set Adaptation
        if lsetadap is None:
            self.own_lsetadap = True
            self.lsetadap = LevelSetMeshAdaptation (mesh, order=order, threshold=0.5,
                                                    discontinuous_qn=True)
        else:
            self.own_lsetadap = False
            self.lsetadap = lsetadap
        
        self.lset_approx = self.lsetadap.lset_p1 
        
        # cut info and XFEM context
        if cutinfo is not None:
            self.own_cutinfo = False
            self.cutinfo = cutinfo
        else:
            self.own_cutinfo = True
            self.cutinfo = CutInfo(mesh)
        

        self.X = VectorH1(mesh, order=order, dgjumps=gp_param!=None)

        self.H = GridFunction(self.X)


    def compute_mean_curvature_vector(self, levelset : CoefficientFunction) -> GridFunction:
        """
        Solve for the mean curvature vector.
        Returns: GridFunction with vector values on the interface.
        """
        if self.own_lsetadap:
            self.lsetadap.CalcDeformation(levelset)
        if self.own_cutinfo:
            self.cutinfo.Update(self.lsetadap.lset_p1)

        u, v = self.X.TnT()
        h = specialcf.mesh_size

        # normal vector from level set
        grad_phi = grad(self.lset_approx)
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
        f += CF((x,y)) * v * ds # InnerProduct(P*E, P*grad(v)) * ds
        f.Assemble()

        # solution
        self.H.vec[:] = 0.0
        self.freedofs = self.X.FreeDofs()
        self.freedofs &= GetDofsOfElements(self.X, self.cutinfo.GetElementsOfType(IF))

        self.H.vec.data = a.mat.Inverse(self.freedofs) * f.vec

    def compute_l2_error(self, H):
        return sqrt(Integrate( InnerProduct(H-self.H,H-self.H) * ds, mesh=self.mesh))

if __name__ == "__main__":
    from netgen.geom2d import SplineGeometry
    from ngsolve import sqrt, x, y, Mesh, specialcf

    geo = SplineGeometry()
    geo.AddRectangle([-2, -2], [2, 2], bc=1)
    mesh = Mesh(geo.GenerateMesh(maxh=0.1))

    phi = sqrt(x*x + y*y) - 1
    h = specialcf.mesh_size
    solver = MeanCurvatureSolver(mesh, order=1, gp_param=0.5*h*h)
    solver.compute_mean_curvature_vector(phi)
    from netgen import gui
    Draw(solver.H, mesh, "H")