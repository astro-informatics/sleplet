import sys
import os
import numpy as np
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht
from sifting_convolution import SiftingConvolution


def dirac_delta_plot(L, resolution, alpha, beta, gamma):
    def fun(l, m):
        return 1
    sc = SiftingConvolution(fun, L, resolution)
    flm = sc.north_pole(m_zero=True)
    f = ssht.inverse(flm, resolution)
    flm_rot = ssht.rotate_flms(
        flm, alpha, beta, gamma, resolution)
    f_rot = ssht.inverse(flm_rot, resolution)
    flm_conv = sc.sifting_convolution(flm, alpha, beta)
    f_conv = ssht.inverse(flm_conv, resolution)
    # self.plotly_plot(f.real)
    # self.plotly_plot(f_rot.real)
    sc.plotly_plot(f_conv.real)

if __name__ == '__main__':
    L = 2 ** 2
    resolution = L * 2 ** 3
    gamma = 0
    beta = np.pi / 4  # theta
    alpha = -np.pi / 4  # phi
    dirac_delta_plot(L, resolution, alpha, beta, gamma)
