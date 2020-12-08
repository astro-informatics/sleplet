import numpy as np
import pyssht as ssht
from numpy.random import default_rng

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import slepian_forward, slepian_inverse
from pys2sleplet.utils.vars import RANDOM_SEED, SAMPLING_SCHEME
from pys2sleplet.utils.wavelet_methods import find_non_zero_wavelet_coefficients


def _signal_power(L: int, signal: np.ndarray) -> float:
    """
    computes the power of the signal
    """
    return (np.abs(signal) ** 2).sum() / L ** 2


def compute_snr(L: int, signal: np.ndarray, noise: np.ndarray) -> float:
    """
    computes the signal to noise ratio
    """
    snr = 10 * np.log10(_signal_power(L, signal) / _signal_power(L, noise))
    signal_type = "Harmonic" if len(signal) == L ** 2 else "Slepian"
    logger.info(f"{signal_type} SNR: {snr:.2f}")
    return snr


def _compute_sigma_noise(L: int, signal: np.ndarray, snr_in: int) -> float:
    """
    compute the std dev of the noise
    """
    return np.sqrt(10 ** (-snr_in / 10) * _signal_power(L, signal))


def create_noise(L: int, signal: np.ndarray, snr_in: int) -> np.ndarray:
    """
    computes Gaussian white noise
    """
    # set random seed
    rng = default_rng(RANDOM_SEED)

    # initialise
    nlm = np.zeros(L ** 2, dtype=np.complex128)

    # std dev of the noise
    sigma_noise = _compute_sigma_noise(L, signal, snr_in)

    # compute noise
    for ell in range(L):
        ind = ssht.elm2ind(ell, 0)
        nlm[ind] = sigma_noise * rng.standard_normal()
        for m in range(1, ell + 1):
            ind_pm = ssht.elm2ind(ell, m)
            ind_nm = ssht.elm2ind(ell, -m)
            nlm[ind_pm] = (
                sigma_noise
                / np.sqrt(2)
                * (rng.standard_normal() + 1j * rng.standard_normal())
            )
            nlm[ind_nm] = (-1) ** m * nlm[ind_pm].conj()
    return nlm


def create_slepian_noise(
    L: int, slepian_signal: np.ndarray, slepian: SlepianFunctions, snr_in: int
) -> np.ndarray:
    """
    computes Gaussian white noise in Slepian space
    """
    flm = ssht.forward(
        slepian_inverse(L, slepian_signal, slepian), L, Method=SAMPLING_SCHEME
    )
    nlm = create_noise(L, flm, snr_in)
    return slepian_forward(L, slepian, flm=nlm)


def _perform_thresholding(
    f: np.ndarray, sigma_j: np.ndarray, n_sigma: int, j: int
) -> np.ndarray:
    """
    set pixels in real space to zero if the magnitude is less than the threshold
    """
    cond = np.abs(f) < n_sigma * sigma_j[j - 1]
    return np.where(cond, 0, f)


def harmonic_hard_thresholding(
    L: int, wav_coeffs: np.ndarray, sigma_j: np.ndarray, n_sigma: int
) -> np.ndarray:
    """
    perform thresholding in harmonic space
    """
    logger.info("begin harmonic hard thresholding")
    for j in range(1, len(wav_coeffs)):
        logger.info(f"start Psi^{j}/{len(wav_coeffs)-1}")
        f = ssht.inverse(wav_coeffs[j], L, Method=SAMPLING_SCHEME)
        f_thresholded = _perform_thresholding(f, sigma_j, n_sigma, j)
        wav_coeffs[j] = ssht.forward(f_thresholded, L, Method=SAMPLING_SCHEME)
    return wav_coeffs


def slepian_hard_thresholding(
    L: int,
    wav_coeffs: np.ndarray,
    sigma_j: np.ndarray,
    n_sigma: int,
    slepian: SlepianFunctions,
) -> np.ndarray:
    """
    perform thresholding in Slepian space
    """
    logger.info("begin Slepian hard thresholding")
    within_shannon_coeffs = find_non_zero_wavelet_coefficients(wav_coeffs)
    for j in range(1, len(within_shannon_coeffs)):
        logger.info(f"start Psi^{j}/{len(within_shannon_coeffs)-1}")
        f = slepian_inverse(L, wav_coeffs[j], slepian)
        f_thresholded = _perform_thresholding(f, sigma_j, n_sigma, j)
        wav_coeffs[j] = slepian_forward(L, slepian, f=f_thresholded)
    return wav_coeffs


def compute_sigma_j(
    L: int, signal: np.ndarray, psi_j: np.ndarray, snr_in: int
) -> np.ndarray:
    """
    compute sigma_j for wavelets used in denoising the signal
    """
    sigma_noise = _compute_sigma_noise(L, signal, snr_in)
    return np.apply_along_axis(
        lambda j: sigma_noise * L * np.sqrt(_signal_power(L, j)), 1, psi_j
    )
