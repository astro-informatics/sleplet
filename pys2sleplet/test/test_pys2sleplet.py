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
    test to ensure that rotation and translation
    give the same result for the Dirac delta
    """
    # setup
    L = ENVS["L"]
    alpha_pi_frac, beta_pi_frac = 0.75, 0.125

    # rotation
    dd = DiracDelta(L)
    dd.rotate(alpha_pi_frac, beta_pi_frac)
    flm_rot = dd.multipole
    f_rot = dd.field

    # translation
    dd = DiracDelta(L)
    dd.translate(alpha_pi_frac, beta_pi_frac)
    flm_trans = dd.multipole
    f_trans = dd.field

    # calculate difference
    flm_diff = flm_rot - flm_trans
    f_diff = f_rot - f_trans

    # perform test
    np.testing.assert_allclose(flm_rot, flm_trans, atol=1e-14)
    np.testing.assert_allclose(f_rot, f_trans, rtol=1e-5)
    print("Translation/rotation difference max error:", np.max(np.abs(flm_diff)))

    # filename
    filename = f"{dd.name}_L{L}_diff_rot_trans_res{dd.resolution}"

    # create plot
    Plot(f_diff.real, dd.resolution, filename).execute()


def test_earth_identity_convolution() -> None:
    """
    test to ensure that the convolving with the
    identity function doesn't change the map
    """
    # setup
    L = ENVS["L"]
    f = Earth(L)
    g = Identity(L)
    flm = f.multipole

    # convolution
    f.convolve(g)
    flm_conv = f.multipole

    # perform test
    np.testing.assert_equal(flm, flm_conv)
    print("Identity convolution passed test")


def test_earth_harmonic_gaussian_convolution() -> None:
    """
    test to ensure that convolving the Earth with the harmonic
    Gausian does not change significantly change the map
    """
    # setup
    L = ENVS["L"]
    f = Earth(L)
    g = HarmonicGaussian(L)
    flm = f.multipole
    f_map = f.field

    # convolution
    f.convolve(g)
    flm_conv = f.multipole
    f_conv = f.field

    # calculate difference
    flm_diff = flm - flm_conv
    f_diff = f_map - f_conv

    # perform test
    np.testing.assert_allclose(flm, flm_conv, atol=5e1)
    np.testing.assert_allclose(f_map, f_conv, atol=8e2)
    print(
        "Earth/harmonic gaussian convolution difference max error:",
        np.max(np.abs(flm_diff)),
    )

    # filename
    filename = f"{g.name}_L{L}_diff_{f.name}_res{f.resolution}_real"

    # create plot
    Plot(f_diff.real, f.resolution, filename).execute()
