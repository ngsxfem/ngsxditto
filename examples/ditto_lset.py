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
# # Ditto geometry extraction

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
mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
d = levelset_coef_from_polar_graph_image('ditto.png')
from ngsolve.webgui import Draw
Draw(d,mesh,"x")

# %%
try:
    from ngsxditto.visualization import PyVistaVisualizer
    with PyVistaVisualizer(mesh, d, "lset", subdivision=5) as viz:
        viz.visualize(scalar_name="temperature", clip_value=0.1)
except:
    pass

