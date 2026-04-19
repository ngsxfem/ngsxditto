# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Computing a Stokes problem

# %% [markdown]
# On the unit square $\Omega = [0,1]^2\subset\mathbb{R}^2$, we want to solve the following problem: $$\begin{align} 
# \text{Find } (u,p) \in [H^1_{D}(\Omega)]^d \times H^1(\Omega) & \text{ so that } \\
# -\Delta u + \nabla p &= f &&\text{ in }\Omega \\ 
# \text{div}\,u &= 0 && \text{ in }\Omega \\
# u &= \left( \begin{array}{cc} 4 y (1-y), & 0 \end{array} \right)^T && \text{ on } \Gamma_{\text{in}} = \{0\} \times [0,1]
# \end{align}$$
#
# where the space $[H^1_D(\Omega)]^d$ already incorporates the Dirichlet boundary conditions on $\Gamma_{\text{in}}$.

# %% [markdown]
# ### Import libraries

# %%
# ngsolve is the underlying finite element library
from ngsolve import *
# our module
from ngsxditto import *

# %% [markdown]
# ### Construct the geometry

# %%
maxh = 0.1
mesh = Mesh(unit_square.GenerateMesh(maxh=maxh))

from ngsolve.webgui import Draw
Draw(mesh)

# %% [markdown]
# ### Construct the fluids discretization

# %% [markdown]
# In `ngsxditto` you have the possibility to construct different types of discretizations for your fluid.
# We will use an $H^1$-conforming discretization which uses the Taylor-Hood element.
# This results in the velocity space $V_h = [\mathbb{P}^k(\mathcal{T}_h)]^d$ and the according pressure space $Q_h = \mathbb{P}^{k-1}(\mathcal{T}_h)$.
# Note that for polynomial orders $k<4$ you do not necessarily get a stable discretization on general meshes.
# Therefore, we will use as a guiding example just $k=4$.

# %%
order = 4

# %% [markdown]
# Now we will implement our discretized fluid.
# First, we will create an object encapsulating the fluid's parameters, like viscosity, density, surface tension coefficient, ...

# %%
fluid_params = FluidParameters(viscosity=1)

# %% [markdown]
# Now we want to actually set up our fluid and its boundary condition.
# To this end, we will first define a `NGSolve`-`CoefficientFunction` describing the boundary conditions.

# %%
uin = CF((4*y*(1-y),0)) # parabolic inflow

fluid = TaylorHood(mesh, order=order, fluid_params=fluid_params, f=CF((8, 0)), add_number_space=True)
fluid.SetOuterBoundaryCondition(StrongDirichletBC(region="left|bottom|top", values=uin))
fluid.InitializeSpaces()
# %% [markdown]
# Next up, we want to initialize and assemble the variational formulation.
# To this end, we need to also pass the right hand side.
# In our case, we pass $ f = (8,0)^T$ which corresponds to a flow from the left inflow to the right side.

# %%
fluid.UpdateActiveDofs()
fluid.InitializeForms()

# %% [markdown]
# Alternatively we can combine above steps by writing: \
# `fluid.Initialize(dirichlet=dirichlet, neumann={}, rhs=rhs)`

# %% [markdown]
# Finally, we can look solve the Stokes problem and take a look at the solution.

# %%
gfup = fluid.SolveStokes()
gfu, gfp, _ = gfup.components
Draw(gfu, mesh)

# %% [markdown]
# You can also prescribe Neumann boundary conditions, describing a stress onto the fluid.
# We will now consider the same domain $\Omega$, but with different boundary conditions: $$\begin{align}
#     u_{D} &= 0 &&\text{ on } \{ 0,1 \} \times [0,1] \\
#     \nabla_{\boldsymbol{n}} u &= -4x(1-x) &&\text{ on } [0,1] \times \{0\} \\
#     \nabla_{\boldsymbol{n}} u &= 4x(1-x) &&\text{ on } [0,1] \times \{1\} 
# \end{align}$$

# %%
uD = CF((0,0))
stress = IfPos(y-0.5, CF((4*y*(1-y),0)), CF((-4*y*(1-y),0)))

# %% [markdown]
# This corresponds to a zero velocity on the top and bottom wall, a suction on the left side of the square, an ejection on the right hand side.
# We can now update our boundary conditions to realize the explained behaviour and assemble the linear system.

# %%
fluid = TaylorHood(mesh, order=order, fluid_params=fluid_params, f=CF((0, 8)), add_number_space=True)
fluid.SetOuterBoundaryCondition(StrongDirichletBC(region="left|right", values=uD))
fluid.SetOuterBoundaryCondition(StrongNeumannBC(region="top|bottom", values=stress))
fluid.Initialize()
# %% [markdown]
# Finally, we can solve the system.

# %%
gfup = fluid.SolveStokes()
gfu, gfp, _ = gfup.components
Draw(gfu)

# %%
