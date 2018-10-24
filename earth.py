import sys
import os
import numpy as np
import scipy.io as sio
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht
from sifting_convolution import SiftingConvolution


def earth():
    '''
    get the flm of the Earth from matlab file
    
    Returns:
        array -- the Earth flm
    '''

    matfile = os.path.join(
        os.environ['SSHT'], 'src', 'matlab', 'data', 'EGM2008_Topography_flms_L0128')
    mat_contents = sio.loadmat(matfile)
    flm = np.ascontiguousarray(mat_contents['flm'][:, 0])

    return flm


def single_plot(L, resolution, alpha, beta, f_type='std', gamma=0):
    flm = earth()
    sc = SiftingConvolution(L, resolution)

    if f_type == 'real':
        f = ssht.inverse(flm, resolution)
        sc.plotly_plot(f.real)
    elif f_type == 'rot':
        flm_rot = ssht.rotate_flms(flm, alpha, beta, gamma, resolution)
        f_rot = ssht.inverse(flm_rot, resolution)
        sc.plotly_plot(f_rot.real)
    else:
        flm_conv = sc.sifting_convolution(flm, alpha, beta)
        f_conv = ssht.inverse(flm_conv, resolution)
        sc.plotly_plot(f_conv.real)
    

def multi_plot(L, resolution, alphas, betas):
    flm = earth()
    sc = SiftingConvolution(L, resolution)
    sc.animation(flm, alphas, betas)


if __name__ == '__main__':
    # single plot
    L = 2 ** 4  # don't use 5 for Earth
    resolution = L * 2 ** 3
    alpha = -np.pi / 4  # phi
    beta = np.pi / 4  # theta
    single_plot(L, resolution, alpha, beta, f_type='std')

    # multi plot
    L = 2 ** 2
    resolution = L * 2 ** 3
    # alphas = np.linspace(-np.pi, np.pi, 17)
    alphas = np.linspace(-np.pi, -np.pi, 1)
    betas = np.linspace(0, np.pi, 9)
    # betas = np.linspace(np.pi, np.pi, 1)
    # multi_plot(L, resolution, alphas, betas)
