#!/usr/bin/env python
from sifting_convolution import SiftingConvolution
from plotting import dirac_delta, earth, read_yaml
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht


def test_dirac_delta_rotate_translate():
    # setup
    flm, config = dirac_delta()
    config['routine'], config['type'] = None, 'real'
    sc = SiftingConvolution(flm, config)
    alpha_pi_fraction, beta_pi_fraction = 0.75, 0.25

    # translate to grid point
    alpha, beta = sc.calc_nearest_grid_point(
        alpha_pi_fraction, beta_pi_fraction)

    # place dirac delta on north pole
    flm_north = sc.place_flm_on_north_pole(flm)

    # rotation
    flm_rot = sc.rotation(flm_north, alpha, beta, gamma=0)
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
    filename = 'dirac_delta_L-' + str(sc.L) + '_diff_rot_trans_samp-' + str(
        sc.method) + '_res-' + str(sc.resolution)

    # create plot
    sc.plotly_plot(f_diff, filename, config['save_fig'])


def test_earth_identity_convolution():
    # get Earth flm
    glm, config = earth()

    # create identity function for test
    def identity(glm):
        # setup
        yaml = read_yaml()
        extra = dict(
            func_name='identity',
            inverted=False,
            reality=False
        )
        config = {**yaml, **extra}
        L = config['L']

        # create identity
        flm = np.ones((L * L)) + 1j * np.zeros((L * L))

        return flm, config

    # setup
    flm, config = identity(glm)
    config['routine'], config['type'] = None, 'real'
    sc = SiftingConvolution(flm, config, earth)

    # convolution
    # conjugate in convolution flips map so
    # need to flip back such that flms are equal for test
    # but need flipped for plot
    flm_conv = sc.convolution(flm, glm)
    flm_conj_conv = np.conj(flm_conv)

    # perform test
    np.testing.assert_equal(flm_conj_conv, glm)
    print('Identity convolution passed test')

    # prepare
    flm_conv_boost = sc.resolution_boost(flm_conv)
    f_conv = ssht.inverse(flm_conv_boost, sc.resolution,
                          Method=sc.method, Reality=sc.reality)

    # filename
    filename = 'identity_L-' + str(sc.L) + '_convolved_earth_L-' + str(
        sc.L) + '_samp-' + str(sc.method) + '_res-' + str(
            sc.resolution) + '_' + config['type']

    # create plot
    sc.plotly_plot(f_conv.real, filename, config['save_fig'])


if __name__ == '__main__':
    test_dirac_delta_rotate_translate()
    test_earth_identity_convolution()
