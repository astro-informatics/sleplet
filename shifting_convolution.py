import sys, os
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
import scipy.io as sio

sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht

class ShiftingConvolution:
    def __init__(self, L, m, gamma, beta, alpha):
        self.L = L
        self.m = m
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

    # Generate spherical harmonics.
    def dirac_delta(self):
        flm = np.zeros((self.L * self.L), dtype=complex)
        flm_np = self.north_pole(flm)

        return flm_np

    # Generate spherical harmonics.
    def random_flm(self):
        flm = np.random.randn(self.L * self.L) +\
              1j * np.random.randn(self.L * self.L)
        flm_np = self.north_pole(flm)

        return flm_np

    def north_pole(self, flm):
        for el in range(self.L):
            ind = ssht.elm2ind(el, self.m)
            north_pole = np.sqrt((2 * el + 1) / (4 * np.pi))
            flm[ind] = north_pole * (1.0 + 1j * 0.0)

        return flm

    # Compute function on the sphere.
    def func_on_spher(self, flm):
        f = ssht.inverse(flm, self.L)

        return f

    # Rotate spherical harmonic
    def rotate(self, flm):
        flm_rot = ssht.rotate_flms(
            flm, self.alpha, self.beta, self.gamma, self.L)

        return flm_rot

    def rot_plot(self, f, f_rot, filename):
        f_plot, mask_array, f_plot_imag, mask_array_imag = ssht.mollweide_projection(
            f, self.L, resolution=200, rot=[0.0, np.pi, np.pi])

        plt.figure()
        plt.subplot(1, 2, 1)
        imgplot = plt.imshow(f_plot, interpolation='nearest')
        plt.colorbar(imgplot, fraction=0.025, pad=0.04)
        plt.imshow(mask_array, interpolation='nearest', cmap=cm.gray, vmin=-1., vmax=1.)
        plt.gca().set_aspect("equal")
        plt.title("f")
        plt.axis('off')

        f_plot, mask_array, f_plot_imag, mask_array_imag = ssht.mollweide_projection(
            f_rot, self.L, resolution=200, rot=[0.0, np.pi, np.pi])

        plt.subplot(1, 2, 2)
        imgplot = plt.imshow(f_plot, interpolation='nearest')
        plt.colorbar(imgplot, fraction=0.025, pad=0.04)
        plt.imshow(mask_array, interpolation='nearest', cmap=cm.gray, vmin=-1., vmax=1.)
        plt.gca().set_aspect("equal")
        plt.title("f rot")
        plt.axis('off')

        plt.savefig(filename + '.png', bbox_inches='tight')
        plt.show()

    def dirac_delta_plot(self):
        flm = self.dirac_delta()
        f = self.func_on_spher(flm)
        flm_rot = self.rotate(flm)
        f_rot = self.func_on_spher(flm_rot)
        sc.rot_plot(f, f_rot, 'diracdelta')

    def random_func_plot(self):
        flm = self.random_flm()
        f = self.func_on_spher(flm)
        flm_rot = self.rotate(flm)
        f_rot = self.func_on_spher(flm_rot)
        sc.rot_plot(f, f_rot, 'randomfunc')


if __name__ == '__main__':
    # Define parameters.
    L = 64
    m = 0
    gamma = 0
    beta = -np.pi / 2
    alpha = 0

    sc = ShiftingConvolution(L,m,gamma,beta,alpha)
    sc.dirac_delta_plot()
    sc.random_func_plot()