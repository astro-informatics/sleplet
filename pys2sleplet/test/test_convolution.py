import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from pys2sleplet.functions.flm.earth import Earth
from pys2sleplet.functions.flm.harmonic_gaussian import HarmonicGaussian
from pys2sleplet.functions.flm.identity import Identity
from pys2sleplet.test.constants import L_LARGE


def test_earth_identity_convolution() -> None:
    """
    test to ensure that the convolving with the
    identity function doesn't change the map
    """
    f = Earth(L_LARGE)
    g = Identity(L_LARGE)
    flm = f.coefficients
    flm_conv = f.convolve(flm, g.coefficients)
    assert_array_equal(flm, flm_conv)


def test_earth_harmonic_gaussian_convolution() -> None:
    """
    test to ensure that convolving the Earth with the harmonic
    Gausian does not change significantly change the map
    """
    f = Earth(L_LARGE)
    g = HarmonicGaussian(L_LARGE)
    flm = f.coefficients
    flm_conv = f.convolve(flm, g.coefficients)
    assert_allclose(np.abs(flm_conv - flm).mean(), 0, atol=7)
