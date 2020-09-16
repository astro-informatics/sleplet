import numpy as np
import pyssht as ssht
from numpy.testing import assert_array_equal, assert_array_less, assert_raises

from pys2sleplet.flm.kernels.axisymmetric_wavelets import AxisymmetricWavelets
from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.test.constants import J_MIN
from pys2sleplet.test.constants import L_LARGE as L
from pys2sleplet.test.constants import N_SIGMA, B
from pys2sleplet.utils.noise import compute_sigma_j, compute_snr, hard_thresholding


def test_denoising_earth_axisymmetric_wavelets() -> None:
    """
    tests that hard thresholding improves the SNR over the map
    """
    # create Earth & noised Earth
    earth = Earth(L)
    earth_noised = Earth(L, noise=True)

    # create wavelets
    aw = AxisymmetricWavelets(L, B=B, j_min=J_MIN)

    # compute wavelet noise
    sigma_j = compute_sigma_j(L, earth.multipole, aw.wavelets[1:])

    # compute wavelet coefficients
    w = np.zeros(aw.wavelets.shape, dtype=np.complex128)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * aw.wavelets[:, ind_m0].conj()
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            w[:, ind] = wav_0 * earth_noised.multipole[ind]

    # hard thresholding
    w_denoised = hard_thresholding(L, w, sigma_j, N_SIGMA)

    # wavelet synthesis
    flm = np.zeros(L ** 2, dtype=np.complex128)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * aw.wavelets[:, ind_m0]
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = (w_denoised[:, ind] * wav_0).sum()

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
