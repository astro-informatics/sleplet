import sys
import os
import numpy as np
import yaml
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht
from sifting_convolution import SiftingConvolution


def setup():
    __location__ = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    yaml_file = os.path.join(__location__, 'input.yml')
    content = open(yaml_file)
    config_dict = yaml.load(content)
    return config_dict


def dirac_delta():
    config = setup()
    L = config['L']
    resolution = L * 2 ** 3
    flm = np.zeros((resolution * resolution), dtype=complex)

    for ell in range(L):
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = 1
    return flm

if __name__ == '__main__':
    sc = SiftingConvolution(dirac_delta, setup())
    sc.plot()
