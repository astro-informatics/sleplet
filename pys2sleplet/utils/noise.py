from typing import Optional

import numpy as np
import pyssht as ssht
from numpy.random import default_rng

from pys2sleplet.utils.integration_methods import (
    calc_integration_resolution,
    calc_integration_weight,
    integrate_sphere,
)
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.mask_methods import create_mask_region
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.vars import RANDOM_SEED


def _signal_power(
    L: int, flm: np.ndarray, region: Optional[Region] = None
) -> np.ndarray:
    """
    computes the power of the harmonic signal
    """
    if region is None:
        integral = (np.abs(flm) ** 2).sum()
    else:
        resolution = calc_integration_resolution(L)
        weight = calc_integration_weight(resolution)
        mask = create_mask_region(resolution, region)
        integral = np.abs(
            integrate_sphere(
                L, resolution, flm, flm, weight, mask_boosted=mask, glm_conj=True
            )
        )
    return integral / L ** 2


def compute_snr(
    L: int, signal: np.ndarray, noise: np.ndarray, region: Optional[Region] = None
) -> float:
    """
    computes the signal to noise ratio
    """
    snr = 10 * np.log10(
        _signal_power(L, signal, region=region) / _signal_power(L, noise, region=region)
    )
    logger.info(f"Noise SNR {'region' if region is not None else ''}: {snr:.2f}")
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
