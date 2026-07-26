from ngsxditto.redistancing import *
from ngsolve import *
from netgen.geom2d import SplineGeometry

domain = SplineGeometry()
domain.AddCircle((0, 0), 1)
mesh = Mesh(domain.GenerateMesh(maxh=0.1))
true_signed_distance = (x ** 2 + y ** 2) ** (1 / 2) - 1 / 2
domain_size = Integrate(CF(1), mesh)
rough = true_signed_distance + 0.08 * sin(15 * x) * cos(13 * y) + 0.04 * sin(29 * x) * sin(23 * y)


def test_does_not_solve_eikonal_alone():
    # explicitly NOT a redistancer: it never iterates towards |grad(phi)|=1,
    # so unlike MinimizationBasedRedistancing (which drives this residual
    # down close to 0, see test_minimization_based_redistancing.py) it must
    # leave a large gradient residual even on its own output. It also is NOT
    # guaranteed to reduce the L2 error against the true SDF when used
    # alone -- its diffusion/mass-anchoring tradeoff can distort the field
    # in ways plain averaging doesn't undo (that's exactly why it must only
    # ever be used as an `initializer=` pre-step, never as the final result).
    order = 2
    phi = GridFunction(H1(mesh, order=order))
    phi.Set(rough)
    GPExtensionRedistancing().Redistance(phi)
    grad_rms = (Integrate((Norm(grad(phi)) - 1) ** 2, mesh) / domain_size) ** (1 / 2)
    assert grad_rms > 0.3


def test_preserves_interface_position():
    order = 2
    phi = GridFunction(H1(mesh, order=order))
    phi.Set(rough)
    GPExtensionRedistancing().Redistance(phi)

    lsetp1 = GridFunction(H1(mesh, order=1))
    InterpolateToP1(phi, lsetp1)
    ci = CutInfo(mesh, lsetp1)
    dS = dCut(levelset=lsetp1, domain_type=IF, definedonelements=ci.GetElementsOfType(IF))
    iface_size = Integrate(CF(1) * dS, mesh)
    assert (Integrate(true_signed_distance ** 2 * dS, mesh) / iface_size) ** (1 / 2) < 5e-2
