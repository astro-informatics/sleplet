from plot_setup import setup
from sifting_convolution import SiftingConvolution
import pyssht as ssht
import sys
import os
import numpy as np
from argparse import ArgumentParser
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))


def extra_setup():
    config, parser = setup()
    parser.add_argument('-l', metavar='ell', type=int, help='multipole')
    parser.add_argument('-m', metavar='m', type=int, help='multipole moment')
    args = parser.parse_args()
    return config, args


def spherical_harmonic():
    config, args = extra_setup()
    L = config['L']
    resolution = config['resolution']

    flm = np.zeros((resolution * resolution), dtype=complex)
    ind = ssht.elm2ind(args.l, args.m)
    flm[ind] = 1
    return flm


if __name__ == '__main__':
    config, args = extra_setup()

    if 'method' not in config:
        config['method'] = args.method
    if 'plotting_type' not in config:
        config['plotting_type'] = args.type

    extra_filename = 'l' + str(args.l) + '_m' + str(args.m) + '_'
    sc = SiftingConvolution(spherical_harmonic, config, extra_filename)
    sc.plot(args.alpha, args.beta, reality=False)
