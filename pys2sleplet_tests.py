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
    alpha, beta = 0.75, 0.25

    # rotation
    flm_rot = sc.rotation(flm, alpha, beta, gamma=0)
    flm_rot_norm = flm_rot / np.linalg.norm(flm_rot)

    # translation
    flm_trans = sc.translation(flm, alpha, beta)
    flm_trans_norm = flm_trans / np.linalg.norm(flm_trans)

    # perform test
    np.testing.assert_allclose(flm_rot_norm, flm_trans_norm, atol=0.15)

if __name__ == '__main__':
    test_dirac_delta_rotate_translate()
