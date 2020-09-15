import numpy as np
import pyssht as ssht

from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.vars import RANDOM_SEED


def _signal_power(L: int, flm: np.ndarray) -> np.ndarray:
    """
    computes the power of the harmonic signal
    """
    return (np.abs(flm) ** 2).sum() / L ** 2


def compute_snr(L: int, signal: np.ndarray, noise: np.ndarray) -> None:
    """
    computes the signal to noise ratio
    """
    snr = 10 * np.log10(_signal_power(L, signal) / _signal_power(L, noise))
    logger.info(f"Noise SNR: {snr:.2f}")


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
    np.random.seed(RANDOM_SEED)

    # initialise
    nlm = np.zeros(L ** 2, dtype=np.complex128)

    # std dev of the noise
    sigma_noise = _compute_sigma_noise(L, signal)

    # compute noise
    for ell in range(L):
        ind = ssht.elm2ind(ell, 0)
        nlm[ind] = 2 * np.random.uniform() - 1
        for m in range(1, ell + 1):
            ind_pm = ssht.elm2ind(ell, m)
            ind_nm = ssht.elm2ind(ell, -m)
            nlm[ind_pm] = sigma_noise * (
                2 * np.random.uniform() - 1 + 1j * (2 * np.random.uniform() - 1)
            )
            nlm[ind_nm] = (-1) ** m * nlm[ind_pm].conj()

    # compute SNR
    compute_snr(L, signal, nlm)
    return nlm


def hard_thresholding(
    L: int, wav_coeffs: np.ndarray, sigma_j: np.ndarray, n_sigma: int
) -> None:
    """
    set pixels in real space to zero if the magnitude is less than the threshold
    """
    for j in range(1, len(wav_coeffs)):
        f = ssht.inverse(wav_coeffs[j], L)
        cond = np.abs(f) < n_sigma * sigma_j[j - 1]
        f = np.where(cond, 0, f)
        wav_coeffs[j] = ssht.forward(f, L)
    return wav_coeffs


def compute_sigma_j(L: int, flm: np.ndarray, psi_j: np.ndarray) -> np.ndarray:
    """
    compute sigma_j for wavelets used in denoising the signal
    """
    sigma_noise = _compute_sigma_noise(L, flm)
    return np.apply_along_axis(
        lambda p: sigma_noise * L * np.sqrt(_signal_power(L, p)), 1, psi_j
    )
