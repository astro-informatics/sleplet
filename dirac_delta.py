from sifting_convolution import SiftingConvolution
import pyssht as ssht
import sys
import os
import numpy as np
import yaml
from argparse import ArgumentParser
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))


def setup():
    __location__ = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    yaml_file = os.path.join(__location__, 'input.yml')
    content = open(yaml_file)
    config_dict = yaml.load(content)

    parser = ArgumentParser(description='Create SSHT plot')
    parser.add_argument('method', metavar='method', type=str, nargs='?', default='north', const='north', choices=['north', 'rotate', 'translate'], help='plotting method i.e. north', )
    parser.add_argument('type', metavar='type', type=str, nargs='?', default='abs', const='north', choices=['abs', 'real', 'imag'], help='plotting method i.e. real')
    parser.add_argument('--alpha', '-a', metavar='alpha', type=float,
                        default=0.0, help='alpha/phi pi fraction')
    parser.add_argument('--beta', '-b', metavar='beta', type=float,
                        default=0.0, help='beta/theta pi fraction')
    args = parser.parse_args()

    return config_dict, args


def dirac_delta():
    config, _ = setup()
    L = config['L']
    resolution = config['resolution']
    flm = np.zeros((resolution * resolution), dtype=complex)

    # impose reality on flms
    for ell in range(L):
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = 1
    return flm

if __name__ == '__main__':
    config, args = setup()

    if 'method' not in config:
        config['method'] = args.method
    if 'plotting_type' not in config:
        config['plotting_type'] = args.type

    sc = SiftingConvolution(dirac_delta, config)

    if config['method'] != 'north':
        alpha_pi_fraction, beta_pi_fraction = args.alpha, args.beta
        sc.plot(alpha_pi_fraction, beta_pi_fraction)
    else:
        sc.plot()
