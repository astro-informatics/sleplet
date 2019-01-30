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


def dirac_delta():
    config, _ = setup()
    L = config['L']
    resolution = config['resolution']

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

    return flm

if __name__ == '__main__':
    config, args = extra_setup()

    if 'method' not in config:
        config['method'] = args.method
    if 'plotting_type' not in config:
        config['plotting_type'] = args.type

    sc = SiftingConvolution(dirac_delta, config)
    sc.plot(args.alpha, args.beta, reality=True)
