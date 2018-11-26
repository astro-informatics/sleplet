import sys
import os
import numpy as np
import yaml
from argparse import ArgumentParser
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


def gaussian(sig=1):
    config = setup()
    L = config['L']
    resolution = L * 2 ** 3
    flm = np.zeros((resolution * resolution), dtype=complex)

    for ell in range(L):
        ind = ssht.elm2ind(ell, m=0)
        flm[ind] = np.exp(-ell * (ell + 1)) / (2 * sig * sig)
    return flm

if __name__ == '__main__':
    config = setup()
    sc = SiftingConvolution(gaussian, config)

    # add from command line
    if config['method'] != 'north':
        parser = ArgumentParser(description='Create SSHT plot')
        parser.add_argument('alpha', metavar='alpha',
                            type=float, help='alpha/phi pi fraction')
        parser.add_argument('beta', metavar='beta',
                            type=float, help='beta/theta pi fraction')
        args = parser.parse_args()
        alpha_pi_fraction, beta_pi_fraction = args.alpha, args.beta
        sc.plot(alpha_pi_fraction, beta_pi_fraction)
    else:
        sc.plot()
