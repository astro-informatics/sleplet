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
        flm = np.random.randn(self.L * self.L) + \
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

    def dirac_delta_plot(self):
        flm = self.dirac_delta()
        f = self.func_on_spher(flm)
        flm_rot = self.rotate(flm)
        f_rot = self.func_on_spher(flm_rot)
        ssht.plot_sphere(f.real, self.L, Output_File='diracdelta_north.png')
        ssht.plot_sphere(f_rot.real, self.L, Output_File='diracdelta_rot.png')

    def random_func_plot(self):
        flm = self.random_flm()
        f = self.func_on_spher(flm)
        flm_rot = self.rotate(flm)
        f_rot = self.func_on_spher(flm_rot)
        ssht.plot_sphere(f.real, self.L, Output_File='gaussian_north.png')
        ssht.plot_sphere(f_rot.real, self.L, Output_File='gaussian_rot.png')


if __name__ == '__main__':
    # Define parameters.
    L = 64
    m = 0
    gamma = 0
    beta = np.pi / 4  # theta
    alpha = -np.pi / 4  # phi

    sc = ShiftingConvolution(L, m, gamma, beta, alpha)
    sc.dirac_delta_plot()
    # sc.random_func_plot()
