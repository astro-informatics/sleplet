import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

import sleplet

L = 128


def test_earth_identity_convolution() -> None:
    """
    Test to ensure that the convolving with the
    identity function doesn't change the map.
    """
    f = sleplet.functions.Earth(L)
    g = sleplet.functions.Identity(L)
    flm = f.coefficients
    flm_conv = f.convolve(flm, g.coefficients)
    assert_array_equal(flm, flm_conv)


def test_earth_harmonic_gaussian_convolution() -> None:
    """
    Test to ensure that convolving the Earth with the harmonic
    Gausian does not change significantly change the map.
    """
    f = sleplet.functions.Earth(L)
    g = sleplet.functions.HarmonicGaussian(L)
    flm = f.coefficients
    flm_conv = f.convolve(flm, g.coefficients)
    assert_allclose(np.abs(flm_conv - flm).mean(), 0, atol=7)


def test_south_america_slepian_identity_convolution(
    slepian_arbitrary,
    south_america_arbitrary,
) -> None:
    """
    Test to ensure that the convolving with the Slepian
    identity function doesn't change the map in Slepian space.
    """
    f_p = sleplet.slepian_methods.slepian_forward(
        slepian_arbitrary.L,
        slepian_arbitrary,
        flm=south_america_arbitrary.coefficients,
    )
    g = sleplet.functions.SlepianIdentity(
        slepian_arbitrary.L,
        region=slepian_arbitrary.region,
    )
    fp_conv = south_america_arbitrary.convolve(
        f_p,
        g.coefficients,
        shannon=slepian_arbitrary.N,
    )
    assert_array_equal(f_p, fp_conv)
