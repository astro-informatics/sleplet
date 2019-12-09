#!/usr/bin/env python
import numpy as np

from pys2sleplet.flm.kernels.dirac_delta import DiracDelta
from pys2sleplet.flm.kernels.harmonic_gaussian import HarmonicGaussian
from pys2sleplet.flm.kernels.identity import Identity
from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.vars import ENVS


def test_dirac_delta_rotate_translate() -> None:
    """
    """
    # setup
    flm = DiracDelta(ENVS["L"])
    alpha_pi_frac, beta_pi_frac = 0.75, 0.125

    # rotation
    flm_rot = flm.rotate(alpha_pi_frac, beta_pi_frac)
    f_rot = flm_rot.invert()

    # translation
    flm_trans = flm.translate(alpha_pi_frac, beta_pi_frac)
    f_trans = flm_trans.invert()

    # calculate difference
    flm_diff = flm_rot - flm_trans
    f_diff = f_rot - f_trans

    # perform test
    np.testing.assert_allclose(flm_rot, flm_trans, atol=1e-14)
    np.testing.assert_allclose(f_rot, f_trans, rtol=1e-5)
    print("Translation/rotation difference max error:", np.max(np.abs(flm_diff)))

    # filename
    filename = f"{flm.name}_L{ENVS['L']}_diff_rot_trans_res{flm_diff.res}"

    # create plot
    Plot(f_diff, f_diff.res, filename).execute()


def test_earth_identity_convolution() -> None:
    """
    """
    # setup
    flm, glm = Identity(ENVS["L"]), Earth(ENVS["L"])

    # convolution
    flm_conv = flm.convolve(glm)

    # perform test
    np.testing.assert_equal(flm, flm_conv)
    print("Identity convolution passed test")


def test_earth_harmonic_gaussian_convolution() -> None:
    """
    """
    # setup
    flm, glm = HarmonicGaussian(ENVS["L"]), Earth(ENVS["L"])

    # map
    f_map = glm.invert()

    # convolution
    flm_conv = flm.convolve(glm)
    f_conv = flm_conv.invert()

    # calculate difference
    flm_diff = flm - flm_conv
    f_diff = f_map - f_conv

    # perform test
    np.testing.assert_allclose(glm, flm_conv, atol=5e1)
    np.testing.assert_allclose(f_map, f_conv, atol=8e2)
    print(
        "Earth/harmonic gaussian convolution difference max error:",
        np.max(np.abs(flm_diff)),
    )

    # filename
    filename = f"{glm.name}_L{ENVS['L']}_diff_{flm.name}_res{flm_conv.res}_real"

    # create plot
    Plot(f_diff.real, f_diff.res, filename).execute()


if __name__ == "__main__":
    test_dirac_delta_rotate_translate()
    test_earth_identity_convolution()
    test_earth_harmonic_gaussian_convolution()
