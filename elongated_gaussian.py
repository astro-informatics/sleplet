from plot_setup import setup
from sifting_convolution import SiftingConvolution
import pyssht as ssht
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))


def extra_setup():
    config, parser = setup()
    args = parser.parse_args()
    return config, args


def grid_fun(theta, phi, theta_0=0, phi_0=np.pi, theta_sig=0.1, phi_sig=1):
    return np.exp(-(0.5 * ((theta - theta_0) / theta_sig) ** 2 + 0.5 * ((phi - phi_0) / phi_sig) ** 2))


def elongated_gaussian():
    config, _ = setup()
    L = config['L']
    resolution = config['resolution']

    thetas, phis = ssht.sample_positions(resolution, Grid=True)
    f = grid_fun(thetas, phis)
    flm = ssht.forward(f, resolution, Reality=True)

    return flm


if __name__ == '__main__':
    config, args = extra_setup()

    if 'method' not in config:
        config['method'] = args.method
    if 'plotting_type' not in config:
        config['plotting_type'] = args.type

    sc = SiftingConvolution(elongated_gaussian, config)
    sc.plot(args.alpha, args.beta, reality=True)
