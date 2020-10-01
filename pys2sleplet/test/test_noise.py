from numpy.testing import assert_array_equal, assert_array_less, assert_raises

from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.test.constants import J_MIN
from pys2sleplet.test.constants import L_LARGE as L
from pys2sleplet.test.constants import N_SIGMA, SNR_IN, B
from pys2sleplet.utils.denoising import denoising_axisym


def test_denoising_earth_axisymmetric_wavelets() -> None:
    """
    tests that hard thresholding improves the SNR over the map
    """
    _, noised_snr, denoised_snr = denoising_axisym(
        "earth", L, B, J_MIN, N_SIGMA, SNR_IN
    )
    assert_array_less(noised_snr, denoised_snr)


def test_adding_noise_changes_flm() -> None:
    """
    tests the addition of Gaussian noise changes the multipole
    """
    earth = Earth(L)
    earth_noised = Earth(L, noise=SNR_IN)
    assert_raises(
        AssertionError, assert_array_equal, earth.multipole, earth_noised.multipole
    )
