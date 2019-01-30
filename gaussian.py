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


def grid_fun(theta, theta_0=0, theta_sig=1):
    return np.exp(-0.5 * ((theta - theta_0) / theta_sig) ** 2)


def gaussian():
    config, _ = setup()
    L = config['L']
    resolution = config['resolution']

    # thetas, _ = ssht.sample_positions(resolution, Grid=True)
    # f = grid_fun(thetas)
    # flm = ssht.forward(f, resolution, Reality=True)

    flm = np.zeros((resolution * resolution), dtype=complex)
    for ell in range(L):
        ind = ssht.elm2ind(ell, m=0)
        # Gaussian with sigma=1
        flm[ind] = np.exp(-ell * (ell + 1) / 2)

    return flm


if __name__ == '__main__':
    config, args = extra_setup()

    if 'method' not in config:
        config['method'] = args.method
    if 'plotting_type' not in config:
        config['plotting_type'] = args.type

    sc = SiftingConvolution(gaussian, config)
    sc.plot(args.alpha, args.beta, reality=True)
