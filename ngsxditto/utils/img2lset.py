import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fft import fft, ifft
from scipy.ndimage import center_of_mass as com
from skimage import measure

from ngsolve import *

def levelset_coef_from_polar_graph_image(image_path: str, n_coefficients: int = 15, center_of_mass: list[float, float] = [0.5, 0.5]):
    """
    Compute the levelset coefficients from a polar graph image.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    n_coefficients : int, optional
        Number of Fourier coefficients to use. Defaults to 15.
    center_of_mass : list[float, float], optional
        Center of mass of the shape. Defaults to [0.5, 0.5].

    Returns
    -------
    d : CoefficientFunction
        Levelset function.
    """
    img = Image.open(image_path).convert('L')
    img = np.array(img)
    binary = (img < 128).astype(np.uint8)  # Schwarz=1, Weiß=0

    M = min(binary.shape)

    # === Mittelpunkt der Figur berechnen ===
    cy, cx = com(binary)

    # === Kontur extrahieren ===
    contours = measure.find_contours(binary, level=0.5)
    # Wir nehmen die längste Kontur
    contour = max(contours, key=len)

    # === Polarkoordinaten berechnen (r, θ) ===
    x_ = contour[:, 1] - cx
    y_ = contour[:, 0] - cy
    theta = np.arctan2(y_, x_)
    r = np.hypot(x_, y_)

    # Sortieren nach θ für sinnvolle Fourier-Analyse
    theta_sorted_idx = np.argsort(theta)
    theta = theta[theta_sorted_idx]
    r = r[theta_sorted_idx]

    # === Fourier-Analyse ===
    N = len(r)
    r_fft = fft(r)
    r_reconstructed = ifft(r_fft).real

    # === Plot ===
    #plt.figure(figsize=(10, 5))

    # Originale Kurve (in Polarkoordinaten)
    #plt.subplot(1, 2, 1, polar=True)
    #plt.plot(theta, r, label='Original')
    #plt.title('Original r(θ)')

    # Rekonstruierte Kurve (nur erste n Fourier-Koeffizienten)
    n = n_coefficients # 15  # Anzahl der Koeffizienten
    r_fft_filtered = np.zeros_like(r_fft)
    r_fft_filtered[:n] = r_fft[:n]
    r_fft_filtered[-n+1:] = r_fft[-n+1:]
    r_smooth = ifft(r_fft_filtered).real
    #r_fft_filtered

    # plt.subplot(1, 2, 2, polar=True)
    # plt.plot(theta, r_smooth, label='Fourier Fit (n=10)')
    # plt.title('Fourier-Fit r(θ)')
    # 
    # plt.tight_layout()
    # plt.show()
    # 
    # plt.subplot(1, 2, 2)
    # plt.plot(theta, r_smooth, label='Fourier Fit (n=10)')
    # plt.title('Fourier-Fit r(θ)')
    # 
    # plt.tight_layout()
    # plt.show()

    c = r_fft / len(r_fft)  # Normalize
    a0 = c[0].real / M / 1.2
    ak = 2 * c[1:n].real / M / 1.2
    bk = -2 * c[1:n].imag / M / 1.2

    # print(f"a0 = {a0}")
    # for k in range(1, n):
    #     print(f"a{k} = {ak[k-1]:.4f}, b{k} = {bk[k-1]:.4f}")

    # from numpy import linspace, pi, cos, sin, exp
    # x = linspace(-pi,pi,100)
    # y = a0 + sum([ak[k-1] * cos(k*x) + bk[k-1] * sin(k*x) for k in range(n-1)]).real
    # plt.subplot(1, 2, 2)
    # plt.plot(x, y, label='asdf')
    # plt.title('Fourier-Fit r(θ)')
    # 
    # plt.tight_layout()
    # plt.show()

    #lset = x
    X = CF((center_of_mass[0]-x,y-center_of_mass[1]))
    theta = IfPos(X[0],atan(X[1]/X[0]),atan(X[1]/X[0])+pi)

    R = a0 + sum([ak[k-1] * cos(k*theta) + bk[k-1] * sin(k*theta) for k in range(n-1)]).real

    d = sqrt(X[0]**2+X[1]**2) - R
    d.Compile()

    return d


if __name__ == '__main__':
    mesh = Mesh(unit_square.GenerateMesh(maxh=0.1))
    d = levelset_coef_from_polar_graph_image('ditto.png')
    from netgen import gui
    Draw(d,mesh,"x")