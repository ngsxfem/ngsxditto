# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mean Curvature ditto

# %%
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Bild einlesen (falls noch nicht geschehen)
image_path = 'ditto.png'  # Passe den Pfad an
img = Image.open(image_path).convert('L')
img_array = np.array(img)

# Bild anzeigen
plt.figure(figsize=(4, 4))
plt.imshow(img_array, cmap='gray')
plt.title("Original BMP")
plt.axis('off')
plt.show()

# %%
from ngsolve import *
from ngsxditto.utils.img2lset import levelset_coef_from_polar_graph_image
mesh = Mesh(unit_square.GenerateMesh(maxh=0.025))
phi = levelset_coef_from_polar_graph_image('ditto.png')
from ngsolve.webgui import Draw
Draw(phi, mesh,"x")

# %%
from netgen.geom2d import SplineGeometry
from ngsolve import sqrt, x, y, Mesh, specialcf
from ngsxditto.fluid.meancurv import MeanCurvatureSolver
from ngsxditto import LevelSetGeometry

lsetgeom = LevelSetGeometry.from_cf(phi,mesh)
mcsolver = MeanCurvatureSolver(mesh, order=1, lset=lsetgeom, gp_param=specialcf.mesh_size)
mcsolver.Step()
Draw(mcsolver.H, mesh, "H")
