from ngsolve import *
from ngsxditto.levelset import *
from ngsxditto.stepper import *
from ngsxditto import direct_solver_spd, direct_solver_nonspd
from xfem import *
from xfem.utils import AdjacencyMatrix, AddNeighborhood
import ngsolve.webgui as ngw

import logging
logger = logging.getLogger(__name__)


class ExtensionOperator(BaseMatrix):
    """
    LinearOperator for the ghost penalty extension. Requires an extension with
    members 
    * inverse
    * mat
    """
    def __init__ (self, ebe):
        super().__init__()
        self.ebe = ebe

    def Mult (self, x, y):
        if not self.ebe.initialized:
            logger.warning("ExtensionOperator not initialized in operator application. Calling Step() of extension.")
            self.ebe.Step()
        y.data = x
        y -= self.ebe.inverse @ self.ebe.mat * x

    def Shape (self):
        return (self.ebe.fes.ndof, self.ebe.fes.ndof)



class ElementBasedExtension(StatelessStepper):
    """
    Generates a linear operator that extends a vector of a field from a submesh 
    to another submesh using  a harmonic extension based on 
    a ghost-penalty(-like) bilinear form which yields a smooth extensions.
    """
    def __init__(self, mesh : Mesh, supportelems : BitArray, targetelems : BitArray, 
                 fes : FESpace, deformation=None,
                 dirichlet_dofs : BitArray = None,
                 energyform = None, activeelems : BitArray = None,
                 activefacets : BitArray = None):
        """
        Initialise the ghost penalty extension with the given parameters.

        Parameters:
        -----------
        mesh: Mesh
            Mesh on which the extension is defined.
        supportelems: BitArray
            BitArray defining the support element submesh for the extension.
            The function to be extended has meaningful values on this submesh **before**
            the extension step.
        targetelems: BitArray
            BitArray defining the target element submesh for the extension.
            The function to be extended has meaningful values on this submesh **after**
            the extension step.
        fes: FESpace
            Finite element space for the extension.
        deformation: GridFunction | None
            Deformation of the mesh (in case of parametric mapping) for the Ghost penalty form.
            Note that this field is ignored in case of custom energy form.
        dirichlet_dofs: BitArray | None
            BitArray defining additional support dofs (that will not be changed)
        energyform: SumOfIntegrals | None
            Energy form for the extension. The default is a standard Ghost Penalty form.
            Experienced user can provide a custom energy form alongside the
            BitArrays `activeelems` and `activefacets`.
        activeelems: BitArray | None
            BitArray defining the active element submesh for the extension.
            These need to be provided if and only if energyform is not None.
            Note that the user needs to update this manually.
        activefacets: BitArray | None
            BitArray defining the active facet submesh for the extension.
            These need to be provided if and only if energyform is not None.
            Note that the user needs to update this manually.
        """
        super().__init__()
        self.mesh = mesh
        self.supportelems = supportelems
        self.targetelems = targetelems

        self.fes = fes

        self.filtered_target = BitArray(mesh.ne)
        self.filtered_support = BitArray(mesh.ne)
        self.target_dofs = BitArray(self.fes.ndof)
        self.support_dofs = BitArray(self.fes.ndof)

        if energyform is None:
            self.activefacets = BitArray(mesh.nfacet)
            self.activeelems = BitArray(mesh.ne)
            self.activeelems.Clear()
            u,v = self.fes.TnT()
            self.energyform = (u - u.Other()) * (v - v.Other()) * dFacetPatch(definedonelements=self.activefacets,
                                                                               deformation=deformation) # Ghost penalty extension
            self.customenergyform = False
            self.filtered_support_or_target = BitArray(mesh.ne)
            self.adjacency = AdjacencyMatrix(mesh, "face")
        else:
            self.energyform = energyform
            self.customenergyform = True
            if activefacets is None:
                self.activefacets = BitArray(mesh.nfacet)
                self.activefacets.Clear()
            else:
                self.activefacets = activefacets
            if activeelems is None:
                self.activeelems = BitArray(mesh.ne)
                self.activeelems.Clear()
            else:
                self.activeelems = activeelems
            if activefacets is None and activeelems is None:
                raise Exception("Either activefacets or activeelems must be provided if energyform is not None!")
        self.dirichlet_dofs = dirichlet_dofs

        self.blf = RestrictedBilinearForm(self.fes, symmetric=not self.customenergyform, 
                                          facet_restriction=self.activefacets, 
                                          element_restriction=self.activeelems, check_unused=False)
        self.blf += self.energyform
        self.operator = ExtensionOperator(self)
        self.initialized = False

    def Step(self):
        """
        Solves for the field on the target domain.
        """

        if not self.customenergyform:
            self.filtered_target[:] = self.targetelems & ~self.supportelems
            self.filtered_support[:] = self.targetelems
            AddNeighborhood(self.filtered_support, self.adjacency, layers=1, inplace=True)
            self.filtered_support &= self.supportelems
            self.filtered_support_or_target[:] = self.filtered_support | self.filtered_target
            self.activefacets[:] = GetFacetsWithNeighborTypes(self.mesh, a= self.filtered_support_or_target, 
                                                              b= self.filtered_support_or_target,
                                                              bnd_val_a=False, bnd_val_b=False, use_and=True)


        else:
            self.filtered_target[:] = self.targetelems
            self.filtered_support[:] = self.supportelems

        self.support_dofs[:] = GetDofsOfElements(self.fes, self.filtered_support)
        if self.dirichlet_dofs is not None:
            self.support_dofs |= self.dirichlet_dofs
        self.target_dofs[:] = GetDofsOfElements(self.fes, self.filtered_target)
        self.target_dofs &= ~self.support_dofs

        self.blf.Assemble(reallocate=True)

        self.mat = self.blf.mat
        if self.customenergyform:
            self.inverse = self.blf.mat.Inverse(freedofs=self.target_dofs, inverse=direct_solver_nonspd)
        else:
            self.inverse = self.blf.mat.Inverse(freedofs=self.target_dofs, inverse=direct_solver_spd)

        self.initialized = True
        return self