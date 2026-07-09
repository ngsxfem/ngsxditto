from netgen.libngpy._meshing import Mesh

from ngsxditto import LevelSetGeometry
from ngsxditto.redistancing import *
from ngsolve import *
from netgen.geom2d import SplineGeometry
from xfem.lsetcurv import *


domain = SplineGeometry()
domain.AddCircle((0,0), 1)
mesh = Mesh(domain.GenerateMesh(maxh=0.1))
circle = x**2 + y**2 - 0.25
true_signed_distance = (x**2 + y**2)**(1/2) - 1/2
domain_size = Integrate(CF(1), mesh)

def test_low_order():
    order = 1

    redistancing = MinimizationBasedRedistancing(alpha=10000)
    levelset = LevelSetGeometry.from_cf(circle, mesh, order=order)
    levelset.SetRedistancing(redistancing)
    levelset.Redistance()
    d_hasif = dx(definedonelements=levelset.cutinfo.GetElementsOfType(IF))
    hasif_size = Integrate(CF(1) * d_hasif, mesh)


    assert 1/domain_size * Integrate((levelset.field - true_signed_distance)**2, mesh)**(1/2) < 1e-2  # L2-error
    assert 1/hasif_size * Integrate((levelset.field - true_signed_distance)**2 * d_hasif, mesh)**(1/2) < 1e-2  # hasif l2 error
    assert Integrate(true_signed_distance**2 * levelset.dS, mesh)**(1/2) < 1e-2  # check preserves interface
    assert Integrate((Norm(grad(levelset.field)) - CF(1))**2, mesh)**(1/2) < 1e-1  # gradient error

def test_high_order():
    order = 3

    redistancing = MinimizationBasedRedistancing(alpha=10000, n_iter=10)
    levelset = LevelSetGeometry.from_cf(circle, mesh, order=order)
    levelset.SetRedistancing(redistancing)
    levelset.Redistance()
    d_hasif = dx(definedonelements=levelset.cutinfo.GetElementsOfType(IF))
    hasif_size = Integrate(CF(1) * d_hasif, mesh)

    assert 1/domain_size * Integrate((levelset.field - true_signed_distance)**2, mesh)**(1/2) < 1e-3  # L2-error
    assert 1/hasif_size * Integrate((levelset.field - true_signed_distance)**2 * d_hasif, mesh)**(1/2) < 1e-3  # hasif l2 error
    assert Integrate(true_signed_distance**2 * levelset.dS, mesh)**(1/2) < 1e-5  # interface error
    assert 1/hasif_size * Integrate((Norm(grad(levelset.field)) - CF(1))**2 * d_hasif, mesh)**(1/2) < 1e-2  # gradient error near interface

