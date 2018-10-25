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


if __name__ == '__main__':
    # initialise class
    L = 2 ** 5
    resolution = L * 2 ** 3
    sc = SiftingConvolution(L, resolution, gaussian)

    # apply rotation/translation
    alpha = -np.pi / 4  # phi
    beta = np.pi / 4  # theta

    sc.fun_plot(alpha, beta)  # north pole
    # sc.fun_plot(alpha, beta, 'rotate')  # rotate
    # sc.fun_plot(alpha, beta, 'translate')  # translate
