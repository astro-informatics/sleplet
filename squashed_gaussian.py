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
    parser.add_argument('method', metavar='method', type=str, nargs='?', default='north', const='north', choices=[
                        'north', 'rotate', 'translate'], help='plotting method i.e. north')
    parser.add_argument('type', metavar='type', type=str, nargs='?', default='real', const='real', choices=[
                        'abs', 'real', 'imag', 'sum'], help='plotting method i.e. real')
    parser.add_argument('--alpha', '-a', metavar='alpha', type=float,
                        default=0.0, help='alpha/phi pi fraction')
    parser.add_argument('--beta', '-b', metavar='beta', type=float,
                        default=0.0, help='beta/theta pi fraction')
    args = parser.parse_args()

    return config_dict, args


def grid_fun(theta, phi, theta_0=0, theta_sig=1):
    return np.exp(-0.5 * ((theta - theta_0) / theta_sig) ** 2) * np.sin(phi)


def squashed_gaussian():
    config, args = setup()
    L = config['L']
    resolution = config['resolution']

    thetas, phis = ssht.sample_positions(resolution, Grid=True)
    f = grid_fun(thetas, phis)
    flm = ssht.forward(f, resolution, Reality=True)

    return flm


if __name__ == '__main__':
    config, args = setup()

    if 'method' not in config:
        config['method'] = args.method
    if 'plotting_type' not in config:
        config['plotting_type'] = args.type

    sc = SiftingConvolution(squashed_gaussian, config)
    sc.plot(args.alpha, args.beta, reality=True)
