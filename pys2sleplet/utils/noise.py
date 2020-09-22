import numpy as np
import pyssht as ssht
from numpy.random import default_rng

from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import slepian_forward, slepian_inverse
from pys2sleplet.utils.vars import RANDOM_SEED


def _signal_power(L: int, signal: np.ndarray) -> np.ndarray:
    """
    computes the power of the signal
    """
    return (np.abs(signal) ** 2).sum() / L ** 2


def compute_snr(L: int, signal: np.ndarray, noise: np.ndarray) -> float:
    """
    computes the signal to noise ratio
    """
    snr = 10 * np.log10(_signal_power(L, signal) / _signal_power(L, noise))
    logger.info(f"Noise SNR: {snr:.2f}")
    return snr


def _compute_sigma_noise(L: int, signal: np.ndarray, snr_in: float = 10) -> float:
    """
    compute the std dev of the noise
    """
    return np.sqrt(10 ** (-snr_in / 10) * _signal_power(L, signal))


def create_noise(L: int, signal: np.ndarray) -> np.ndarray:
    """
    computes Gaussian white noise
    """
    # set random seed
    rng = default_rng(RANDOM_SEED)

    # initialise
    nlm = np.zeros(L ** 2, dtype=np.complex128)

    # std dev of the noise
    sigma_noise = _compute_sigma_noise(L, signal)

    # compute noise
    for ell in range(L):
        ind = ssht.elm2ind(ell, 0)
        nlm[ind] = rng.uniform(-1, 1)
        for m in range(1, ell + 1):
            ind_pm = ssht.elm2ind(ell, m)
            ind_nm = ssht.elm2ind(ell, -m)
            nlm[ind_pm] = sigma_noise * (rng.uniform(-1, 1) + 1j * rng.uniform(-1, 1))
            nlm[ind_nm] = (-1) ** m * nlm[ind_pm].conj()
    return nlm


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
    for j in range(1, len(wav_coeffs)):
        f = ssht.inverse(wav_coeffs[j], L)
        f_thresholded = _perform_thresholding(f, sigma_j, n_sigma, j)
        wav_coeffs[j] = ssht.forward(f_thresholded, L)
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
    for j in range(1, len(wav_coeffs)):
        f = slepian_inverse(L, wav_coeffs[j], slepian)
        f_thresholded = _perform_thresholding(f, sigma_j, n_sigma, j)
        flm = ssht.forward(f_thresholded, L)
        wav_coeffs[j] = slepian_forward(L, flm, slepian)
    return wav_coeffs


def compute_sigma_j(L: int, signal: np.ndarray, psi_j: np.ndarray) -> np.ndarray:
    """
    compute sigma_j for wavelets used in denoising the signal
    """
    sigma_noise = _compute_sigma_noise(L, signal)
    return np.apply_along_axis(
        lambda j: sigma_noise * L * np.sqrt(_signal_power(L, j)), 1, psi_j
    )
