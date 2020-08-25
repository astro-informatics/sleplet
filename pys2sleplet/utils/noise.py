import numpy as np
import pyssht as ssht

from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.vars import RANDOM_SEED


def _signal_power(L: int, flm: np.ndarray) -> np.ndarray:
    """
    computes the power of the harmonic signal
    """
    return (np.abs(flm) ** 2).sum() / L ** 2


def _compute_snr(L: int, signal: np.ndarray, noise: np.ndarray) -> float:
    """
    computes the signal to noise ratio
    """
    return 10 * np.log10(_signal_power(L, signal) / _signal_power(L, noise))


def create_noise(L: int, signal: np.ndarray, snr_in: float = 10) -> np.ndarray:
    """
    computes Gaussian white noise
    """
    # set random seed
    np.random.seed(RANDOM_SEED)

    # initialise
    nlm = np.zeros(L ** 2, dtype=np.complex128)

    # std dev of the noise
    sigmanoise = np.sqrt(10 ** (-snr_in / 10) * _signal_power(L, signal))

    # compute noise
    for ell in range(L):
        ind = ssht.elm2ind(ell, 0)
        nlm[ind] = 2 * np.random.uniform() - 1
        for m in range(1, ell + 1):
            ind_pm = ssht.elm2ind(ell, m)
            ind_nm = ssht.elm2ind(ell, -m)
            nlm[ind_pm] = sigmanoise * (
                2 * np.random.uniform() - 1
            ) + 1j * sigmanoise * (2 * np.random.uniform() - 1)
            nlm[ind_nm] = (-1) ** m * nlm[ind_pm].conj()

    # compute SNR
    snr = _compute_snr(L, signal, nlm)
    logger.info(f"Noise applied - SNR: {snr:.2f}")
    return nlm
