from numpy.testing import assert_array_equal, assert_array_less, assert_raises

from pys2sleplet.flm.kernels.axisymmetric_wavelets import AxisymmetricWavelets
from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.test.constants import J_MIN
from pys2sleplet.test.constants import L_LARGE as L
from pys2sleplet.test.constants import N_SIGMA, B
from pys2sleplet.utils.noise import compute_sigma_j, compute_snr, hard_thresholding
from pys2sleplet.utils.wavelet_methods import (
    axisymmetric_wavelet_forward,
    axisymmetric_wavelet_inverse,
)


def test_denoising_earth_axisymmetric_wavelets() -> None:
    """
    tests that hard thresholding improves the SNR over the map
    """
    # create Earth & noised Earth
    earth = Earth(L)
    earth_noised = Earth(L, noise=True)

    # create wavelets
    aw = AxisymmetricWavelets(L, B=B, j_min=J_MIN)

    # compute wavelet coefficients
    w = axisymmetric_wavelet_forward(L, earth_noised.multipole, aw.wavelets)

    # compute wavelet noise
    sigma_j = compute_sigma_j(L, earth.multipole, aw.wavelets[1:])

    # hard thresholding
    w_denoised = hard_thresholding(L, w, sigma_j, N_SIGMA)

    # wavelet synthesis
    flm = axisymmetric_wavelet_inverse(L, w_denoised, aw.wavelets)

    # compute SNR
    noised = compute_snr(L, earth.multipole, earth_noised.multipole - earth.multipole)
    denoised = compute_snr(L, earth.multipole, flm - earth.multipole)

    assert_array_less(noised, denoised)


def test_adding_noise_changes_flm() -> None:
    """
    tests the addition of Gaussian noise changes the multipole
    """
    earth = Earth(L)
    earth_noised = Earth(L, noise=True)
    assert_raises(
        AssertionError, assert_array_equal, earth.multipole, earth_noised.multipole
    )
