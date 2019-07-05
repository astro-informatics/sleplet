#!/usr/bin/env python
from letter import dirac_delta, earth, identity
from plotting import Plotting
from sifting_convolution import SiftingConvolution
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.environ["SSHT"], "src", "python"))
import pyssht as ssht


def test_dirac_delta_rotate_translate() -> None:
    # setup
    flm, name, config = dirac_delta()
    config["routine"], config["type"], config["annotation"] = None, None, False
    sc = SiftingConvolution(flm, name, config)
    sc.calc_nearest_grid_point(alpha_pi_fraction=0.75, beta_pi_fraction=0.25)
    plotting = Plotting(
        method=sc.method, auto_open=config["auto_open"], save_fig=config["save_fig"]
    )

    # rotation
    flm_rot = sc.rotation(flm, sc.alpha, sc.beta, gamma=0)
    flm_rot_boost = plotting.resolution_boost(flm_rot, sc.L, sc.resolution)
    f_rot = ssht.inverse(
        flm_rot_boost, sc.resolution, Method=sc.method, Reality=sc.reality
    )

    # translation
    flm_trans = sc.translation(flm)
    flm_trans_boost = plotting.resolution_boost(flm_trans, sc.L, sc.resolution)
    f_trans = ssht.inverse(
        flm_trans_boost, sc.resolution, Method=sc.method, Reality=sc.reality
    )

    # calculate difference
    flm_diff = flm_rot - flm_trans
    f_diff = f_rot - f_trans

    # perform test
    np.testing.assert_allclose(flm_rot, flm_trans, atol=1e-14)
    np.testing.assert_allclose(f_rot, f_trans, rtol=1e-5)
    print("Translation/rotation difference max error:", np.max(np.abs(flm_diff)))

    # filename
    filename = (
        f"dirac_delta_L-{sc.L}_diff_rot_trans_samp-{sc.method}_res-{sc.resolution}"
    )

    # create plot
    plotting.plotly_plot(f_diff, filename)


def test_earth_identity_convolution() -> None:
    # setup
    flm, flm_name, config = earth()
    glm, glm_name, _ = identity()
    config["routine"], config["type"], config["annotation"] = None, None, False
    sc = SiftingConvolution(flm, flm_name, config, glm, glm_name)
    sc.calc_nearest_grid_point()
    plotting = Plotting(
        method=sc.method, auto_open=config["auto_open"], save_fig=config["save_fig"]
    )

    # convolution
    flm_conv = sc.convolution(flm, glm)

    # perform test
    np.testing.assert_equal(flm_conv, flm)
    print("Identity convolution passed test")

    # prepare
    flm_conv_boost = plotting.resolution_boost(flm_conv, sc.L, sc.resolution)
    f_conv = ssht.inverse(
        flm_conv_boost, sc.resolution, Method=sc.method, Reality=sc.reality
    )

    # filename
    filename = f"identity_L-{sc.L}_convolved_earth_L-{sc.L}_samp-{sc.method}_res-{sc.resolution}_real"

    # create plot
    plotting.plotly_plot(f_conv.real, filename)


if __name__ == "__main__":
    test_dirac_delta_rotate_translate()
    test_earth_identity_convolution()
