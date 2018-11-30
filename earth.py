import sys
import os
import numpy as np
import yaml
import scipy.io as sio
from sifting_convolution import SiftingConvolution


def setup():
    __location__ = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    yaml_file = os.path.join(__location__, 'earth.yml')
    content = open(yaml_file)
    config_dict = yaml.load(content)
    return config_dict


def earth():
    matfile = os.path.join(
        os.environ['SSHT'], 'src', 'matlab', 'data', 'EGM2008_Topography_flms_L0128')
    mat_contents = sio.loadmat(matfile)
    flm = np.ascontiguousarray(mat_contents['flm'][:, 0])
    return flm

if __name__ == '__main__':
    sc = SiftingConvolution(earth, setup())
    sc.plot()
