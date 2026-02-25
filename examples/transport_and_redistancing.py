# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Levelset Transport and Redistancing

# %%
from PIL import Image
import numpy as np
from ngsolve import *
from xfem import *

from ngsxditto.utils.img2lset import levelset_coef_from_polar_graph_image
from netgen.geom2d import SplineGeometry
import ngsolve.webgui as ngw
from ngsxditto.gradient_tester import NaiveGradientTester
from ngsxditto.transport import ImplicitSUPGTransport, ExplicitDGTransport
from ngsxditto.levelset import *

# %%
image_path = 'ditto.png'  # Passe den Pfad an
img = Image.open(image_path).convert('L')
img_array = np.array(img)

domain = SplineGeometry()
domain.AddCircle((0,0), 1)

mesh = Mesh(domain.GenerateMesh(maxh=0.1))
phi_start = levelset_coef_from_polar_graph_image('ditto.png', center_of_mass=[0., -0.5])
ngw.Draw(phi_start, mesh, "x")

# %%
wind = CF((-y, x))
dt = 0.01
t = Parameter(0)

transport = ImplicitSUPGTransport(mesh, wind, order=2, inflow_values=None, dt=dt)
transport.SetInitialValues(phi_start)
transport.time = t

redistancing = FastMarching()
levelset = LevelSetGeometry(transport, redistancing)

# %% [markdown]
# You can propagate the levelset function the following way. This will solve the linear using the system defined by the `transport` object.

# %%
levelset.RunUntilTime(pi)
ngw.Draw(levelset.transport.field)

# %% [markdown]
# You can change the velocity field and the time-step size using the `SetWind` and the `SetTimeStepSize` method. This will automatically reassemble the linear system if necessary.

# %%
levelset.transport.SetWind(-wind)
levelset.transport.SetTimeStepSize(0.02)
levelset.RunFixedSteps(50)
ngw.Draw(levelset.transport.field)

# %% [markdown]
# Over time the levelset function can deform, since the numerical errors accumulate. To prevent this from effecting the interface we can apply redistancing to convert the levelset function to an approximate signed distance function w.r.t. the interface.

# %%
levelset.Redistance()
ngw.Draw(levelset.field)

# %% [markdown]
# We can automatically apply redistancing by defining an autoredistancing scheme. When propagating the levelset function this will automatically check if the criterion for redistancing is fulfilled and apply the redistancing algorithm given to the `LevelSetGeometry`. In the following code redistancing will be applied if the norm of the gradient is out of the bounds `(0.5, 2)` at any point.

# %%
t = Parameter(0)
dt = 0.01

transport = ExplicitDGTransport(mesh, wind, order=1, inflow_values=None, dt=dt, compile=False)
transport.SetInitialValues(phi_start)
transport.time = t
redistancing = FastMarching()
gradient_tester = NaiveGradientTester(mesh)
autoredistancing = GradientRedistancing(gradient_tester=gradient_tester, gradient_bounds=(0.6, 1.5))
levelset = LevelSetGeometry(transport, redistancing, autoredistancing=autoredistancing)
levelset.multistepper.RunUntilTime(4*pi)
ngw.Draw(levelset.field)

# %%
