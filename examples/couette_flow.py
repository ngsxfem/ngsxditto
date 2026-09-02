# # Couette Flow

from ngsxditto.utils.loglevel import loggingSlider
loggingSlider(default_level="WARNING") # global level
loggingSlider("ngsxditto", default_level="DEBUG") # ngsxditto

from ngsxditto import *
from ngsolve import *
from xfem import *
import ngsolve.webgui as ngw
from netgen.occ import *
from netgen.geom2d import *

geo = SplineGeometry()
L = 2
H = L/10
geo.AddRectangle((-L/2, -H/2), (L/2, H/2), bcs=["bottom", "right", "top", "left"])
mesh = Mesh(geo.GenerateMesh(maxh=0.05))
ngw.Draw(mesh)

# +
order = 2
dt = 1e-2

u_top = -0.1
u_bottom = 0.1
u_in_out = CF((u_bottom + (u_top - u_bottom) * (y + H/2) / H, 0))

contact_angle = 1/3 * pi
# -

t = Parameter(0)
starting_levelset = CF(x)
transport = ExplicitDGTransport(mesh, dt=dt, order=order, compile=False)
redistancing = MinimizationBasedRedistancing(initializer=FastMarching())
autoredistancing = PeriodicRedistancing(20)
levelset = LevelSetGeometry(transport, redistancing=redistancing, autoredistancing=autoredistancing,
                            boundary_tangential="bottom|top")
levelset.Initialize(starting_levelset)

# +
fluid1_params = FluidParameters(viscosity=1e-2)
fluid2_params = FluidParameters(viscosity=1e-1)

wall_params = WallParameters(region="bottom|top", contact_angle=contact_angle, friction_coeff_surface=10,
                             friction_coeff_line=0,
                            wall_velocities={"bottom": CF((u_bottom, 0)), "top": CF((u_top, 0))})
mean_curvature = MeanCurvatureSolver(mesh, order=order, lset=levelset)
mean_curvature.Step()
fluid = TwoPhaseTaylorHood(mesh, fluid1_params=fluid1_params, fluid2_params=fluid2_params, lset=levelset,
                           nitsche_stab=100, f1=CF((0, 0)), f2=CF((0, 0)), surface_tension_coeff=0.05,
                           surface_tension=mean_curvature.H, dt=dt, order=order + 1, ghost_stab=1e-2,
                           advection=True, time_order=1, extension_radius=0.1,
                           wall_params=wall_params)
fluid.SetOuterBoundaryCondition(StrongDirichletBC("left|right", u_in_out))
fluid.SetOuterBoundaryCondition(StrongNormalVelocityBC("top|bottom"))

fluid.Initialize()
sol = fluid.SolveStokes()
gfu, gfp = sol.components[0], sol.components[1]
u1, u2 = gfu.components
p1, p2 = gfp.components
fluid.Initialize(initial_velocity1=u1, initial_velocity2=u2)

DrawDC(levelset.field, u1, u2, mesh)

# +
velocity_extension = LevelsetBasedExtension(levelset, order=order, gamma=1e-1, ghost_stab=1, no_penetration="top|bottom", no_slip="")

velocity_extension.SetRhs(fluid.gfu.components[0])
levelset.transport.SetWind(velocity_extension.field)

def should_finalize():
    return time_loop.i_inner == 3

end_time = 1

def debug():
    ngw.Draw(velocity_extension.field)
    ngw.Draw(levelset.field)
    ngw.Draw(IfPos(levelset.field, fluid.gfu.components[1], fluid.gfu.components[0]), mesh)

time_loop = TimeLoop(time=t, dt=dt, end_time=end_time, display_progress_bar=True, should_finalize=None)
time_loop.SetFinalizeRule(should_finalize)

cf_neg = Norm(fluid.gfu.components[0])
cf_pos = Norm(fluid.gfu.components[1])
animation = UnfittedNGSWebguiPlot(levelset, cf_neg=cf_neg, cf_pos=cf_pos,
                                  order=fluid.order, time=t, end_time=end_time,
                                  name="animation", min=0, max=0.2, autoscale=False)

time_loop.Register(velocity_extension, name="vel ext.")
time_loop.Register(levelset, name="levelset")
time_loop.Register(mean_curvature, name="mean curvature")
time_loop.Register(fluid, name="moving stokes")
time_loop.Register(animation, name="animation")
#time_loop.Register(debug, as_validate=True)

time_loop()
# -

DrawDC(levelset.field, 1, 0, mesh)


