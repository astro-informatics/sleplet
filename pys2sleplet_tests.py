#!/usr/bin/env python
from sifting_convolution import SiftingConvolution
from plotting import dirac_delta, earth, identity, read_yaml
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht


def test_dirac_delta_rotate_translate():
    # setup
    flm, name, config = dirac_delta()
    config['routine'], config['type'] = None, None
    sc = SiftingConvolution(flm, name, config)
    alpha_pi_fraction, beta_pi_fraction = 0.75, 0.25
    sc.calc_nearest_grid_point(alpha_pi_fraction, beta_pi_fraction)

    # rotation
    flm_rot = sc.rotation(flm, sc.alpha, sc.beta, gamma=0)
    flm_rot_boost = sc.resolution_boost(flm_rot)
    f_rot = ssht.inverse(flm_rot_boost, sc.resolution,
                         Method=sc.method, Reality=sc.reality)

    # translation
    flm_trans = sc.translation(flm)
    flm_trans_boost = sc.resolution_boost(flm_trans)
    f_trans = ssht.inverse(flm_trans_boost, sc.resolution,
                           Method=sc.method, Reality=sc.reality)

    # calculate difference
    flm_diff = flm_rot - flm_trans
    f_diff = f_rot - f_trans

    # perform test
    np.testing.assert_allclose(flm_rot, flm_trans, atol=1e-14)
    np.testing.assert_allclose(f_rot, f_trans, rtol=1e-6)
    print('Translation/rotation difference max error:',
          np.max(np.abs(flm_rot - flm_trans)))

    # filename
    filename = (f'dirac_delta_L-{sc.L}_diff_rot_trans_samp-'
                f'{sc.method}_res-{sc.resolution}')

    # create plot
    sc.plotly_plot(f_diff, filename, config['save_fig'])


def test_earth_identity_convolution():
    # setup
    flm, flm_name, config = earth()
    glm, glm_name, _ = identity()
    config['routine'], config['type'] = None, None
    sc = SiftingConvolution(flm, flm_name, config, glm, glm_name)
    sc.calc_nearest_grid_point()

    # convolution
    flm_conv = sc.convolution(flm, glm)

    # perform test
    np.testing.assert_equal(flm_conv, flm)
    print('Identity convolution passed test')

    # prepare
    flm_conv_boost = sc.resolution_boost(flm_conv)
    f_conv = ssht.inverse(flm_conv_boost, sc.resolution,
                          Method=sc.method, Reality=sc.reality)

    # filename
    filename = (f'identity_L-{sc.L}_convolved_earth_L-'
                f'{sc.L}_samp-{sc.method}_res-{sc.resolution}_real')

    # create plot
    sc.plotly_plot(f_conv.real, filename, config['save_fig'])


if __name__ == '__main__':
    test_dirac_delta_rotate_translate()
    test_earth_identity_convolution()
