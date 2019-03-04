#!/usr/bin/env python
from sifting_convolution import SiftingConvolution
from plotting import dirac_delta
import numpy as np
import sys
import os
import pyssht as ssht
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))


def test_dirac_delta_rotate_translate():
    # setup
    flm, config = dirac_delta()
    config.update(dict.fromkeys(['routine', 'type'], None))
    sc = SiftingConvolution(flm, config)
    L, method, reality = config['L'], config['sampling'], config['reality']
    resolution = L * 2 ** config['pow2_res2L']
    alpha, beta = 0.75, 0.25

    # rotation
    flm_rot = sc.rotation(flm, alpha, beta, gamma=0)
    # translation
    flm_trans = sc.translation(flm, alpha, beta)

    # difference plot
    filename = 'difference_dirac_delta_f_rot-f_trans'
    flm_diff = flm_rot - flm_trans
    flm = sc.resolution_boost(flm_diff)
    f = ssht.inverse(flm, resolution, Method=method, Reality=reality)
    sc.plotly_plot(f, filename)

    # perform test
    np.testing.assert_allclose(flm_rot, flm_trans, atol=2.8)

if __name__ == '__main__':
    test_dirac_delta_rotate_translate()
