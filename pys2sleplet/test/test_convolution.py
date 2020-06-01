import numpy as np
from numpy.testing import assert_array_equal

from pys2sleplet.flm.kernels.harmonic_gaussian import HarmonicGaussian
from pys2sleplet.flm.kernels.identity import Identity
from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.test.constants import L_LARGE as L
from pys2sleplet.utils.logger import logger


def test_earth_identity_convolution() -> None:
    """
    test to ensure that the convolving with the
    identity function doesn't change the map
    """
    f = Earth(L)
    g = Identity(L)
    flm = f.multipole

    f.convolve(g.multipole)
    flm_conv = f.multipole

    assert_array_equal(flm, flm_conv)
    logger.info("Identity convolution passed test")


def test_earth_harmonic_gaussian_convolution() -> None:
    """
    test to ensure that convolving the Earth with the harmonic
    Gausian does not change significantly change the map
    """
    f = Earth(L)
    g = HarmonicGaussian(L)
    flm, f_map = f.multipole, f.field

    f.convolve(g.multipole)
    flm_conv, f_conv = f.multipole, f.field

    flm_diff = flm - flm_conv

    assert_array_equal(flm, flm_conv)
    assert_array_equal(f_map, f_conv)
    logger.info(
        "Earth/harmonic gaussian convolution difference max error: "
        f"{np.abs(flm_diff).max()}"
    )
