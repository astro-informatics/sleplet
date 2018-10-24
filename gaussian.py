import sys
import os
import numpy as np
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht
from sifting_convolution import SiftingConvolution


def gaussian(ell, m, sig=1):
    '''
    function to place on the sphere
    
    Arguments:
        ell {int} -- current multipole value
        m {int} -- m <= |ell|
    
    Keyword Arguments:
        sig {int} -- standard deviation (default: {1})
    
    Returns:
        float -- function to pass to SiftingConvolution
    '''

    return np.exp(-ell * (ell + 1)) / (2 * sig * sig)


def single_plot(L, resolution, alpha, beta, f_type='std', gamma=0):
    sc = SiftingConvolution(L, resolution, gaussian)
    flm = sc.north_pole(m_zero=True)

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
    sc = SiftingConvolution(L, resolution, gaussian)
    flm = sc.north_pole(m_zero=True)
    sc.animation(flm, alphas, betas)


if __name__ == '__main__':
    # single plot
    L = 2 ** 5
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
