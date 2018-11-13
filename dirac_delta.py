import sys
import os
import numpy as np
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht
from sifting_convolution import SiftingConvolution


def dirac_delta(ell, m):
    '''
    function to place on the sphere

    Arguments:
        ell {int} -- current multipole value
        m {int} -- m <= |ell|

    Returns:
        float -- function to pass to SiftingConvolution
    '''

    return 1


if __name__ == '__main__':
    # initialise class
    L = 2 ** 5
    resolution = L * 2 ** 3
    sc = SiftingConvolution(L, resolution, dirac_delta)

    # apply rotation/translation
    alpha = 0  # phi
    # alpha = np.pi  # phi
    # alpha = np.pi * 7 / 4
    beta = 0  # theta
    # beta = np.pi  # theta
    # beta = np.pi / 4

    dir = 'figures/'
    auto_open = True
    # north pole
    sc.plot(dir, 0, 0, 'abs', auto_open, 'north')
    sc.plot(dir, 0, 0, 'real', auto_open, 'north')
    sc.plot(dir, 0, 0, 'imag', auto_open, 'north')
    for alpha in [np.pi * x / 4.0 for x in range(9)]:
        for beta in [np.pi * x / 4.0 for x in range(5)]:
            for plot in ['abs', 'real', 'imag']:
                for method in ['rotate', 'translate']:
                    print(alpha, beta, plot, method)
                    sc.plot(dir, alpha, beta, plot, auto_open, method)