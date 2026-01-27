from ngsolve import *
from ngsolve import unit_square, Mesh, CoefficientFunction, CF
from xfem import *

def CheckElementHistory(els_current, els_old_list_temp, **kwargs) -> None:  # von Christoph
	"""
	Checks if all current elements have the necessary history for BDF. Prints error message if some elements do not have the required history for the BDF scheme used.

	### Parameters
		* els_current (BitArray) : Elements which require a history
		* els_old_list_temp (list[BitArray]) : Extension elements at previous steps
	### Returns
		* None

	"""
	if type(els_old_list_temp) == BitArray:
		els_old_list = [els_old_list_temp]
	else:
		els_old_list = els_old_list_temp

	els_test = BitArray(len(els_current))
	for els_old in els_old_list:
		els_test.Clear()
		els_test |= els_current & (~els_old)
		if els_test.NumSet() > 0:
			print(Red + f"Some active elements do not have a history! - Number of active elements {els_current.NumSet()} thereof {els_test.NumSet()} without history" + Normal_color)
			print("Parameters:")
			for key, value in kwargs.items():
				print(f"{key}: {value}")
			raise Exception
	els_test = None
	return None

class LevelSetTransport:
	def __init__(self, mesh:Mesh, dt, order, old_lset, active_elements:BitArray, wind:CoefficientFunction):
		with TaskManager():
			# copies
			self.dt = dt
			self.mesh = mesh
			self.wind = wind
			if active_elements is None:
				self.active_elements = BitArray(mesh.ne)
				self.active_elements[:] = True
			else:
				self.active_elements = active_elements

			self.old_lset = old_lset
			self.active_facets = BitArray(mesh.nfacet)

			# spaces and gridfunctions
			self.fes = L2(self.mesh, order=order, all_dofs_together=True, dgjumps=True)
			self.fesH1 = H1(mesh, order=order, dgjumps=True)

			self.lset_bnd_gf = GridFunction(self.fesH1, nested=False)                           # dirchitlet bnd values
			self.levelsetH1 = GridFunction(self.fesH1, nested=False)                            # continuous level set function
			self.gfu = GridFunction(self.fes, autoupdate=True)    # DG level set function

			# markers
			self.bnd_facets = BitArray(mesh.nfacet)  # boundary facets of active transport elements
			self.BndElem = BitArray(self.mesh.ne)        # elements at boundary of active transport elements
			self.TransDofs = BitArray(self.fesH1.ndof)  # dofs of active transport elements

			# auxiliary functions
			self.bnd_facets_ind = GridFunction(FacetFESpace(mesh,order=0))	# 1 on Boundary and 0 everywhere else
			self.nobnd_facets_ind = IfPos(self.bnd_facets_ind, 0, 1) 									# faster than nobnd_facets_ind = IndicatorCF(mesh, ~bnd_facets, facets=True)
			self.ProjNotTrans = Projector(self.TransDofs, False)						# project on outside of active transport elements

			if wind is not None:
				self.SetWind(wind)

	def SetWind(self, wind: CoefficientFunction):
		u, v = self.gfu.space.TnT()
		n = specialcf.normal(self.mesh.dim)

		# build linear system
		self.bfa:Bilinearform = RestrictedBilinearForm(self.fes, element_restriction=self.active_elements, facet_restriction=self.active_facets, check_unused=False)
		self.bfa += u * v * dx(definedonelements=self.active_elements)
		self.bfa += self.dt * (v * (wind | grad(u))).Compile() * dx(definedonelements=self.active_elements, bonus_intorder=1) # ! bonus_intorder=1 important, because integration order would be too low otherwise
		self.bfa += self.dt*(- IfPos((wind|n), 0, (wind|n) * (u - self.nobnd_facets_ind *u.Other())) * v).Compile() * dx(element_boundary=True, definedonelements=self.active_elements)

		self.lf = LinearForm(self.fes)
		self.lf += (self.old_lset * v).Compile() * dx(definedonelements=self.active_elements)
		self.lf += -self.dt*(IfPos((self.wind|n), 0, (self.wind|n) * self.old_lset * v * self.bnd_facets_ind)).Compile() * dx(definedonelements=self.BndElem, element_boundary=True, bonus_intorder=1)  # integral über rand facets (mit bnd_facets_ind und nur über BndElems)


	def Step(self):

		# update markers and spaces
		self.TransDofs[:] = GetDofsOfElements(self.fesH1, self.active_elements)

		with TaskManager():
			self.active_facets[:] = GetFacetsWithNeighborTypes(self.mesh, a=self.active_elements, b=self.active_elements, use_and=False) ##todo
			self.bnd_facets[:] = GetFacetsWithNeighborTypes(self.mesh, a=self.active_elements, b=~self.active_elements, bnd_val_a=False, bnd_val_b=True)
			self.BndElem[:] = GetElementsWithNeighborFacets(self.mesh, self.bnd_facets) & self.active_elements

			self.bnd_facets_ind.vec[:] = 0
			self.bnd_facets_ind.vec[self.bnd_facets] = 1

			# solve linear system
			self.bfa.Assemble(reallocate=True)
			self.lf.Assemble()

			# if self.time is not None:
			# 	self.time.Set(self.time.Get() + self.dt)

			freedofs = GetDofsOfElements(self.fes, self.active_elements)
			self.gfu.vec.data = self.bfa.mat.Inverse(freedofs = freedofs, inverse="pardiso") * self.lf.vec

			self.levelsetH1.Set(self.gfu, definedonelements=self.active_elements)
			self.levelsetH1.vec.data += self.ProjNotTrans.CreateSparseMatrix() * self.old_lset.vec # overwrite outside of self.active_elements with old values (no new zero level)