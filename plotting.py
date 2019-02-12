#!/usr/bin/env python
from sifting_convolution import SiftingConvolution
import pyssht as ssht
import sys
import os
import numpy as np
import yaml
from argparse import ArgumentParser
import scipy.io as sio
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))


def read_yaml(yaml_file):
    __location__ = os.path.realpath(os.path.join(
        os.getcwd(), os.path.dirname(__file__)))
    yaml_file = os.path.join(__location__, yaml_file)
    content = open(yaml_file)
    config_dict = yaml.load(content)
    return config_dict


def read_args(spherical_harmonic=False):
    parser = ArgumentParser(description='Create SSHT plot')
    parser.add_argument('flm', type=valid_plotting, choices=list(
        total.keys()), help='flm to plot on the sphere')
    parser.add_argument('--type', '-t', type=str, nargs='?', default='real', const='real', choices=['abs', 'real', 'imag', 'sum'], help='plotting type: defaults to real')
    parser.add_argument('--method', '-m', type=str, nargs='?', default='north', const='north', choices=['north', 'rotate', 'translate'], help='plotting method: defaults to north')
    parser.add_argument('--alpha', '-a', type=float, default=0.0,
                        help='alpha/phi pi fraction - defaults to 0')
    parser.add_argument('--beta', '-b', type=float, default=0.0,
                        help='beta/theta pi fraction - defaults to 0')
    parser.add_argument('--convolve', '-c', type=valid_maps, choices=list(maps.keys(
    )), help='glm to perform sifting convolution with i.e. flm x glm*')

    # extra args for spherical harmonics
    if spherical_harmonic:
        parser.add_argument('-l', metavar='ell', type=int, help='multipole')
        parser.add_argument('-m', metavar='m', type=int,
                            help='multipole moment')

    args = parser.parse_args()
    return args


def dirac_delta():
    # setup
    config = read_yaml('input.yml')
    L, pow2_res2L = config['L'], config['pow2_res2L']
    resolution = L * 2 ** pow2_res2L
    config['func_name'] = 'dirac_delta'

    # create flm
    flm = np.zeros((resolution * resolution), dtype=complex)
    # impose reality on flms
    for ell in range(L):
        m = 0
        ind = ssht.elm2ind(ell, m)
        flm[ind] = 1
        for m in range(1, ell + 1):
            ind_pm = ssht.elm2ind(ell, m)
            ind_nm = ssht.elm2ind(ell, -m)
            flm[ind_pm] = 1
            flm[ind_nm] = (-1) ** m * np.conj(flm[ind_pm])

    return flm, config


def gaussian(sig=10):
    # setup
    config = read_yaml('input.yml')
    L, pow2_res2L = config['L'], config['pow2_res2L']
    resolution = L * 2 ** pow2_res2L
    config['func_name'] = 'gaussian'

    # create flm
    flm = np.zeros((resolution * resolution), dtype=complex)
    for ell in range(L):
        ind = ssht.elm2ind(ell, m=0)
        flm[ind] = np.exp(-ell * (ell + 1) / (2 * sig * sig))

    return flm, config


def squashed_gaussian():
    # setup
    config = read_yaml('input.yml')
    L, pow2_res2L = config['L'], config['pow2_res2L']
    resolution = L * 2 ** pow2_res2L
    config['func_name'] = 'squashed_gaussian'

    # function on the grid
    def fun(theta, phi, theta_0=0, theta_sig=0.1):
        return np.exp(-0.5 * ((theta - theta_0) / theta_sig) ** 2) * np.sin(phi)

    thetas, phis = ssht.sample_positions(resolution, Grid=True)
    f = fun(thetas, phis)
    flm = ssht.forward(f, resolution, Reality=True)

    return flm, config


def elongated_gaussian():
    # setup
    config = read_yaml('input.yml')
    L, pow2_res2L = config['L'], config['pow2_res2L']
    resolution = L * 2 ** pow2_res2L
    config['func_name'] = 'elongated_gaussian'

    # function on the grid
    def fun(theta, phi, theta_0=0, phi_0=np.pi, theta_sig=0.1, phi_sig=1):
        return np.exp(-(0.5 * ((theta - theta_0) / theta_sig) ** 2 + 0.5 * ((phi - phi_0) / phi_sig) ** 2))

    thetas, phis = ssht.sample_positions(resolution, Grid=True)
    f = fun(thetas, phis)
    flm = ssht.forward(f, resolution, Reality=True)

    return flm, config


def spherical_harmonic(ell, m):
    # setup
    config = read_yaml('input.yml')
    L, pow2_res2L = config['L'], config['pow2_res2L']
    resolution = L * 2 ** pow2_res2L
    config['func_name'] = 'spherical_harmonic_l' + str(ell) + '_m' + str(m)
    config['reality'] = False

    # create flm
    flm = np.zeros((resolution * resolution), dtype=complex)
    ind = ssht.elm2ind(ell, m)
    flm[ind] = 1

    return flm, config


def earth():
    # setup
    config = read_yaml('earth.yml')

    # create flm
    matfile = os.path.join(
        os.environ['SSHT'], 'src', 'matlab', 'data', 'EGM2008_Topography_flms_L0128')
    mat_contents = sio.loadmat(matfile)
    flm = np.ascontiguousarray(mat_contents['flm'][:, 0])

    return flm, config


def valid_plotting(func_name):
    # check if valid function
    if func_name in total:
        return func_name
    else:
        raise ValueError('Not a valid function name to plot')


def valid_maps(func_name):
    # check if valid function
    if func_name in maps:
        return func_name
    else:
        raise ValueError('Not a valid map name to convolve')


functions = {
    'dirac_delta': dirac_delta,
    'gaussian': gaussian,
    'squashed_gaussian': squashed_gaussian,
    'elongated_gaussian': elongated_gaussian,
    'spherical_harmonic': spherical_harmonic
}
maps = {
    'earth': earth
}
total = {**functions, **maps}

if __name__ == '__main__':
    if sys.argv[1] == 'spherical_harmonic':
        args = read_args(True)
        flm_input = total[args.flm]
        flm, flm_config = flm_input(args.l, args.m)
    else:
        args = read_args()
        flm_input = total[args.flm]
        flm, flm_config = flm_input()

    if 'method' not in flm_config:
        flm_config['method'] = args.method
    if 'plotting_type' not in flm_config:
        flm_config['plotting_type'] = args.type

    # if convolving function passed otherwise return None
    glm = maps.get(args.convolve)
    sc = SiftingConvolution(flm, flm_config, glm)
    sc.plot(args.alpha, args.beta)
