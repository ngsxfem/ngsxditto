# Instrumented Hysing-Turek case-1 run for the docs benchmark:
#  - dumps the levelset FIELD before/after every redistancing (gallery of the
#    +/- 0.025*i isolines), via a LevelSetGeometry subclass that hooks Redistance;
#  - dumps mean curvature H (+ lsetp1 + deformation) at snapshot times, to colour
#    the zero-isoline by |H|;
#  - counts how often redistancing fires, so PeriodicRedistancing(n) can be
#    compared against a gradient-driven trigger.
#
# env: MAXH DT ORDER OUTDIR ENDTIME TRIGGER(periodic|gradient) REDIST_EVERY
#      GRAD_LO GRAD_HI NTHREADS
import os, math
import numpy as np
from ngsxditto.utils.loglevel import loggingSlider
loggingSlider(default_level="WARNING"); loggingSlider("ngsxditto", default_level="WARNING")
from ngsxditto import *
from ngsolve import *
from xfem import *
from netgen.occ import OCCGeometry, MoveTo, X, Y

maxh   = float(os.environ.get("MAXH", "0.06"))
dt     = float(os.environ.get("DT", "0.01"))
order  = int(os.environ.get("ORDER", "2"))
outdir = os.environ.get("OUTDIR", "probe")
end_time = float(os.environ.get("ENDTIME", "3.0"))
trigger = os.environ.get("TRIGGER", "periodic")
redist_every = int(os.environ.get("REDIST_EVERY", "20"))
grad_lo = float(os.environ.get("GRAD_LO", "0.75"))
grad_hi = float(os.environ.get("GRAD_HI", "1.30"))
nthreads = int(os.environ.get("NTHREADS", "8"))
os.makedirs(outdir, exist_ok=True)
SetNumThreads(nthreads)
tag = trigger

g = 0.98
rho1, mu1, sigma = 100.0, 1.0, 24.5
rho2, mu2        = 1000.0, 10.0
t = Parameter(0.0)


class ProbedLevelSet(LevelSetGeometry):
    """LevelSetGeometry that dumps field before/after each redistancing and counts events."""
    def attach_probe(self, time_param, probe_dir, curvature=None):
        self.time_param = time_param
        self.probe_dir = probe_dir
        self.curvature = curvature
        self.redist_count = 0
        self.redist_times = []

    def _dump(self, name):
        VTKOutput(self.mesh, coefs=[self.field, self.deformation], names=["phi", "deform"],
                  filename=os.path.join(self.probe_dir, name), subdivision=self.order).Do(vb=VOL)

    def Redistance(self):
        k = self.redist_count
        self._dump(f"{tag}_redist{k:02d}_before")
        super().Redistance()
        self._dump(f"{tag}_redist{k:02d}_after")
        self.redist_times.append(round(self.time_param.Get(), 4))
        self.redist_count += 1


# ---- mesh / levelset / redistancing trigger --------------------------------
domain = MoveTo(0, 0).Rectangle(1, 2).Face()
domain.edges.Min(X).name = "left";   domain.edges.Max(X).name = "right"
domain.edges.Min(Y).name = "bottom"; domain.edges.Max(Y).name = "top"
mesh = Mesh(OCCGeometry(domain, dim=2).GenerateMesh(maxh=maxh))

transport = ExplicitDGTransport(mesh, dt=dt, order=order, compile=False)
if trigger == "gradient":
    from ngsxditto.gradient_tester import NaiveGradientTester
    auto = GradientRedistancing(gradient_tester=NaiveGradientTester(mesh),
                                gradient_bounds=(grad_lo, grad_hi))
else:
    auto = PeriodicRedistancing(redist_every)
levelset = ProbedLevelSet(transport, redistancing=MinimizationBasedRedistancing(),
                          autoredistancing=auto)
levelset.Initialize(sqrt((x - 0.5)**2 + (y - 0.5)**2) - 0.25)

f1p = FluidParameters(viscosity=mu1, density=rho1, surface_tension_coeff=sigma)
f2p = FluidParameters(viscosity=mu2, density=rho2)   # surface tension lives on fluid1 only
mc = MeanCurvatureSolver(mesh, order=order, lset=levelset, gp_param=1); mc.Step()
levelset.attach_probe(t, outdir, curvature=mc)
grav = CF((0, -g))
fluid = TwoPhaseTaylorHood(mesh, fluid1_params=f1p, fluid2_params=f2p, lset=levelset,
                           surface_tension=mc.H, f1=grav, f2=grav, dt=dt, order=order + 1,
                           time_order=2, add_convection=True, ghost_stab=1, nitsche_stab=100)
fluid.SetOuterBoundaryCondition(StrongDirichletBC(region="top|bottom", values=CF((0, 0))))
fluid.SetOuterBoundaryCondition(NitscheNormalVelocityBC(region="left|right", values=CF(0)))
fluid.Initialize()

velocity_extension = LevelsetBasedExtension(levelset, gamma=1e-3, order=order)
velocity_extension.SetRhs(fluid.gfu.components[0])
levelset.transport.SetWind(velocity_extension.field)

# ---- curvature snapshots (|H| on the zero isoline) -------------------------
curv_vtk = VTKOutput(mesh, coefs=[levelset.lsetp1, levelset.deformation, mc.H],
                     names=["phi", "deform", "H"], filename=os.path.join(outdir, f"{tag}_curv"),
                     subdivision=order)
snap_times = set(np.round(np.linspace(0, end_time, 9), 4))
def maybe_curv():
    if any(abs(t.Get() - s) < dt/2 for s in snap_times):
        curv_vtk.Do(vb=VOL, time=round(t.Get(), 4))

time_loop = TimeLoop(time=t, dt=dt, end_time=end_time, display_progress_bar=False)
time_loop.SetFinalizeRule(lambda: time_loop.i_inner >= 3)
time_loop.Register(velocity_extension, name="vel ext.")
time_loop.Register(levelset,           name="levelset")
time_loop.Register(mc,                 name="mean curvature")
time_loop.Register(fluid,              name="two-phase flow")
time_loop.Register(FunctionCallStepper(maybe_curv, as_validate=True), name="curv snap")

maybe_curv()
with TaskManager():
    time_loop()

n_steps = round(end_time / dt)
print(f"PROBE {tag}: maxh={maxh} dt={dt} steps={n_steps} redistancings={levelset.redist_count} "
      f"at t={levelset.redist_times}", flush=True)
np.savez(os.path.join(outdir, f"{tag}_meta.npz"),
         redist_count=levelset.redist_count, redist_times=np.array(levelset.redist_times),
         n_steps=n_steps, maxh=maxh, dt=dt, order=order,
         verts=np.array([list(v.point) for v in mesh.vertices]),
         tris=np.array([[v.nr for v in el.vertices] for el in mesh.Elements(VOL)]))
