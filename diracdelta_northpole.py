import sys, os
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
import scipy.io as sio

sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht

# Define parameters.
L = 64
m = 0
gamma = 0
beta = -np.pi / 2
alpha = 0

# Generate spherical harmonics.
flm = np.zeros((L * L), dtype=complex)
for el in range(L):
    ind = ssht.elm2ind(el, m)
    north_pole = np.sqrt((2 * el + 1) / (4 * np.pi))
    flm[ind] = north_pole * (1.0 + 1j * 0.0)

# Compute function on the sphere.
f = ssht.inverse(flm, L)

# Rotate spherical harmonic
flm_rot = ssht.rotate_flms(flm, alpha, beta, gamma, L)

# Compute rotated function on the sphere.
f_rot = ssht.inverse(flm_rot, L)

# Plot
f_plot, mask_array, f_plot_imag, mask_array_imag = ssht.mollweide_projection(f, L, resolution=200,
                                                                             rot=[0.0, np.pi, np.pi])
plt.figure()
plt.subplot(1, 2, 1)
imgplot = plt.imshow(f_plot, interpolation='nearest')
plt.colorbar(imgplot, fraction=0.025, pad=0.04)
plt.imshow(mask_array, interpolation='nearest', cmap=cm.gray, vmin=-1., vmax=1.)
plt.gca().set_aspect("equal")
plt.title("f")
plt.axis('off')

f_plot, mask_array, f_plot_imag, mask_array_imag = ssht.mollweide_projection(f_rot, L, resolution=200,
                                                                             rot=[0.0, np.pi, np.pi])
plt.subplot(1, 2, 2)
imgplot = plt.imshow(f_plot, interpolation='nearest')
plt.colorbar(imgplot, fraction=0.025, pad=0.04)
plt.imshow(mask_array, interpolation='nearest', cmap=cm.gray, vmin=-1., vmax=1.)
plt.gca().set_aspect("equal")
plt.title("f rot")
plt.axis('off')

plt.savefig('diracdelta.png', bbox_inches='tight')
plt.show()
