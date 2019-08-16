#!/usr/bin/env python
from plotting import Plotting
from sifting import dirac_delta, earth, identity, morlet
from sifting_convolution import SiftingConvolution
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.environ["SSHT"], "src", "python"))
import pyssht as ssht


def test_dirac_delta_rotate_translate() -> None:
    # setup
    flm, name, config = dirac_delta()
    sc = SiftingConvolution(flm, name, config)
    sc.calc_nearest_grid_point(alpha_pi_fraction=0.75, beta_pi_fraction=0.25)
    plotting = Plotting(auto_open=sc.auto_open, save_fig=sc.save_fig)

    # rotation
    flm_rot = sc.rotation(flm, sc.alpha, sc.beta, gamma=0)
    flm_rot_boost = plotting.resolution_boost(flm_rot, sc.L, sc.resolution)
    f_rot = ssht.inverse(
        flm_rot_boost, sc.resolution, Reality=sc.reality, Method="MWSS"
    )

    # translation
    flm_trans = sc.translation(flm)
    flm_trans_boost = plotting.resolution_boost(flm_trans, sc.L, sc.resolution)
    f_trans = ssht.inverse(
        flm_trans_boost, sc.resolution, Reality=sc.reality, Method="MWSS"
    )

    # calculate difference
    flm_diff = flm_rot - flm_trans
    f_diff = f_rot - f_trans

    # perform test
    np.testing.assert_allclose(flm_rot, flm_trans, atol=1e-14)
    np.testing.assert_allclose(f_rot, f_trans, rtol=1e-5)
    print("Translation/rotation difference max error:", np.max(np.abs(flm_diff)))

    # filename
    filename = f"dirac_delta_L-{sc.L}_diff_rot_trans_res-{sc.resolution}"

    # create plot
    plotting.plotly_plot(f_diff, sc.resolution, filename, sc.annotations())


def test_earth_identity_convolution() -> None:
    # setup
    flm, flm_name, config = earth()
    glm, glm_name, _ = identity()
    sc = SiftingConvolution(flm, flm_name, config, glm, glm_name)

    # convolution
    flm_conv = sc.convolution(flm, glm)

    # perform test
    np.testing.assert_equal(flm_conv, flm)
    print("Identity convolution passed test")


def test_earth_morlet_convolution() -> None:
    # setup
    flm, flm_name, config = earth()
    glm, glm_name, _ = morlet()
    sc = SiftingConvolution(flm, flm_name, config, glm, glm_name)
    sc.calc_nearest_grid_point()
    plotting = Plotting(auto_open=sc.auto_open, save_fig=sc.save_fig)

    # map
    flm_map_boost = plotting.resolution_boost(flm, sc.L, sc.resolution)
    f_map = ssht.inverse(
        flm_map_boost, sc.resolution, Reality=sc.reality, Method="MWSS"
    )

    # convolution
    flm_conv = sc.convolution(flm, glm)
    flm_conv_boost = plotting.resolution_boost(flm_conv, sc.L, sc.resolution)
    f_conv = ssht.inverse(
        flm_conv_boost, sc.resolution, Reality=sc.reality, Method="MWSS"
    )

    # calculate difference
    flm_diff = flm - flm_conv
    f_diff = f_map - f_conv

    # perform test
    np.testing.assert_allclose(flm, flm_conv, atol=5e1)
    np.testing.assert_allclose(f_map, f_conv, atol=8e2)
    print("Earth/morlet convolution difference max error:", np.max(np.abs(flm_diff)))

    # filename
    filename = f"earth_L-{sc.L}_diff_morlet_res-{sc.resolution}_real"

    # create plot
    plotting.plotly_plot(f_diff.real, sc.resolution, filename, sc.annotations())


if __name__ == "__main__":
    test_dirac_delta_rotate_translate()
    test_earth_identity_convolution()
    test_earth_morlet_convolution()
