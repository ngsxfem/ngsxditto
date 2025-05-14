from ngsxditto.fluid import H1ConformingFluid, FluidParameters
from ngsolve import *


def test_h1conf_import():
    maxh = 0.2
    mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))
    fluid_params = FluidParameters(viscosity=1e-3)
    fluid = H1ConformingFluid(mesh, fluid_params=fluid_params)

if __name__=="__main__":
    test_h1conf_import()