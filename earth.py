import sys
import os
import numpy as np
import scipy.io as sio
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht
from sifting_convolution import SiftingConvolution


def earth(ell=0, m=0):
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


if __name__ == '__main__':
    # initialise class
    L = 2 ** 4  # don't use 5 for Earth
    resolution = L * 2 ** 3
    sc = SiftingConvolution(L, resolution, earth)

    flm = earth()
    f = ssht.inverse(flm, sc.resolution, Reality=True)
    filename = 'figures/earth.html'
    sc.plotly_plot(f, filename)
