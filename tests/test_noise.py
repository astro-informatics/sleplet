from numpy.testing import assert_array_equal, assert_array_less, assert_raises

from sleplet.denoising import denoising_axisym
from sleplet.functions.flm.axisymmetric_wavelets import AxisymmetricWavelets
from sleplet.functions.flm.earth import Earth

B = 2
J_MIN = 0
L = 128
N_SIGMA = 3
SNR_IN = 10


def test_denoising_earth_axisymmetric_wavelets() -> None:
    """
    tests that hard thresholding improves the SNR over the map
    """
    fun = Earth(L)
    fun_noised = Earth(L, noise=SNR_IN)
    aw = AxisymmetricWavelets(L, B=B, j_min=J_MIN)
    _, noised_snr, denoised_snr = denoising_axisym(fun, fun_noised, aw, SNR_IN, N_SIGMA)
    assert isinstance(noised_snr, float)
    assert_array_less(noised_snr, denoised_snr)


def test_adding_noise_changes_flm() -> None:
    """
    tests the addition of Gaussian noise changes the coefficients
    """
    earth = Earth(L)
    earth_noised = Earth(L, noise=SNR_IN)
    assert_raises(
        AssertionError,
        assert_array_equal,
        earth.coefficients,
        earth_noised.coefficients,
    )
