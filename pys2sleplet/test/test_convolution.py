import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from pys2sleplet.functions.flm.earth import Earth
from pys2sleplet.functions.flm.harmonic_gaussian import HarmonicGaussian
from pys2sleplet.functions.flm.identity import Identity
from pys2sleplet.functions.fp.slepian_identity import SlepianIdentity
from pys2sleplet.test.constants import L_LARGE
from pys2sleplet.utils.slepian_methods import slepian_forward


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


def test_south_america_slepian_identity_convolution(
    slepian_arbitrary, south_america_arbitrary
) -> None:
    """
    test to ensure that the convolving with the Slepian
    identity function doesn't change the map in Slepian space
    """
    fp = slepian_forward(
        slepian_arbitrary.L, slepian_arbitrary, flm=south_america_arbitrary.coefficients
    )
    g = SlepianIdentity(slepian_arbitrary.L, region=slepian_arbitrary.region)
    fp_conv = south_america_arbitrary.convolve(
        fp, g.coefficients, shannon=slepian_arbitrary.N
    )
    assert_array_equal(fp, fp_conv)
