from numpy.testing import assert_array_equal, assert_array_less, assert_raises

from sleplet.functions.flm.axisymmetric_wavelets import AxisymmetricWavelets
from sleplet.functions.flm.earth import Earth
from sleplet.test.constants import J_MIN, L_LARGE, N_SIGMA, SNR_IN, B
from sleplet.utils.denoising import denoising_axisym


def test_denoising_earth_axisymmetric_wavelets() -> None:
    """
    tests that hard thresholding improves the SNR over the map
    """
    fun = Earth(L_LARGE)
    fun_noised = Earth(L_LARGE, noise=SNR_IN)
    aw = AxisymmetricWavelets(L_LARGE, B=B, j_min=J_MIN)
    _, noised_snr, denoised_snr = denoising_axisym(fun, fun_noised, aw, SNR_IN, N_SIGMA)
    assert_array_less(noised_snr, denoised_snr)


def test_adding_noise_changes_flm() -> None:
    """
    tests the addition of Gaussian noise changes the coefficients
    """
    earth = Earth(L_LARGE)
    earth_noised = Earth(L_LARGE, noise=SNR_IN)
    assert_raises(
        AssertionError,
        assert_array_equal,
        earth.coefficients,
        earth_noised.coefficients,
    )
