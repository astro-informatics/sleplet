import sys
import os
import numpy as np
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht
from sifting_convolution import SiftingConvolution


def squashed_gaussian(ell, m, sig=1):
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

    return np.exp(m) * np.exp(-ell * (ell + 1)) / (2 * sig * sig)


if __name__ == '__main__':
    # initialise class
    L = 2 ** 5
    resolution = L * 2 ** 3
    sc = SiftingConvolution(L, resolution, squashed_gaussian)

    # apply rotation/translation
    # alpha = 0  # phi
    # alpha = np.pi  # phi
    alpha = np.pi * 7 / 4
    # beta = 0  # theta
    # beta = np.pi  # theta
    beta = np.pi / 4

    dir = os.path.expanduser('~') + '/Dropbox/cosmoinformatics/'
    # sc.plot(dir, alpha, beta, 'imag')  # north pole
    # sc.plot(dir, alpha, beta, 'imag', 'rotate')  # rotate
    sc.plot(dir, alpha, beta, 'real', 'translate')  # translate
