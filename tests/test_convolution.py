import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from sleplet.functions import Earth, HarmonicGaussian, Identity, SlepianIdentity
from sleplet.slepian_methods import slepian_forward

L = 128


def test_earth_identity_convolution() -> None:
    """Test to ensure that the convolving with the
    identity function doesn't change the map.
    """
    f = Earth(L)
    g = Identity(L)
    flm = f.coefficients
    flm_conv = f.convolve(flm, g.coefficients)
    assert_array_equal(flm, flm_conv)


def test_earth_harmonic_gaussian_convolution() -> None:
    """Test to ensure that convolving the Earth with the harmonic
    Gausian does not change significantly change the map.
    """
    f = Earth(L)
    g = HarmonicGaussian(L)
    flm = f.coefficients
    flm_conv = f.convolve(flm, g.coefficients)
    assert_allclose(np.abs(flm_conv - flm).mean(), 0, atol=7)


def test_south_america_slepian_identity_convolution(
    slepian_arbitrary,
    south_america_arbitrary,
) -> None:
    """Test to ensure that the convolving with the Slepian
    identity function doesn't change the map in Slepian space.
    """
    f_p = slepian_forward(
        slepian_arbitrary.L,
        slepian_arbitrary,
        flm=south_america_arbitrary.coefficients,
    )
    g = SlepianIdentity(slepian_arbitrary.L, region=slepian_arbitrary.region)
    fp_conv = south_america_arbitrary.convolve(
        f_p,
        g.coefficients,
        shannon=slepian_arbitrary.N,
    )
    assert_array_equal(f_p, fp_conv)
