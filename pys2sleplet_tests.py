#!/usr/bin/env python
from sifting_convolution import SiftingConvolution
from plotting import dirac_delta
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.environ['SSHT'], 'src', 'python'))
import pyssht as ssht


def test_dirac_delta_rotate_translate(save_fig):
    # setup
    flm, config = dirac_delta()
    config['routine'], config['type'] = None, 'abs'
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
    flm_trans = sc.translation(flm, alpha, beta)
    flm_trans_boost = sc.resolution_boost(flm_trans)
    f_trans = ssht.inverse(flm_trans_boost, sc.resolution,
                           Method=sc.method, Reality=sc.reality)

    # filename
    filename = 'dirac_delta_L-' + str(sc.L) + '_diff_rot_trans_samp-' + str(
        sc.method) + '_res-' + str(sc.resolution) + '_' + str(sc.type)

    # calculate difference
    flm_diff = flm_rot - flm_trans
    f_diff = f_rot - f_trans

    # perform test
    np.testing.assert_allclose(flm_rot, flm_trans, rtol=1e-10)
    np.testing.assert_allclose(f_rot, f_trans, rtol=1e-9)

    # create plot
    sc.plotly_plot(f_diff, filename, save_fig)


if __name__ == '__main__':
    test_dirac_delta_rotate_translate(save_fig=False)
