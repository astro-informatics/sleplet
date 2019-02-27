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
    alpha, beta = 0.75, 0.25

    # rotation
    flm_rot = sc.rotation(flm, alpha, beta, gamma=0)
    flm_rot_norm = flm_rot / np.linalg.norm(flm_rot)
    f_rot = ssht.inverse(flm_rot, L, Method=method, Reality=reality)
    f_rot_norm = ssht.inverse(flm_rot_norm, L, Method=method, Reality=reality)

    # translation
    flm_trans = sc.translation(flm, alpha, beta)
    flm_trans_norm = flm_trans / np.linalg.norm(flm_trans)
    f_trans = ssht.inverse(flm_trans, L, Method=method, Reality=reality)
    f_trans_norm = ssht.inverse(flm_trans_norm, L, Method=method, Reality=reality)

    # perform test
    np.testing.assert_allclose(flm_rot, flm_trans, atol=2.5)
    np.testing.assert_allclose(flm_rot_norm, flm_trans_norm, atol=0.14)
    np.testing.assert_allclose(f_rot, f_trans, rtol=1e5)
    np.testing.assert_allclose(f_rot_norm, f_trans_norm, atol=7)

if __name__ == '__main__':
    test_dirac_delta_rotate_translate()
