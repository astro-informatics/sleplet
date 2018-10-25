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
    alpha = -np.pi / 4  # phi
    beta = np.pi / 4  # theta

    sc.fun_plot(alpha, beta)  # north pole
    # sc.fun_plot(alpha, beta, 'rotate')  # rotate
    # sc.fun_plot(alpha, beta, 'translate')  # translate
