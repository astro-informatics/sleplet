from typing import Optional

import numpy as np
import pyssht as ssht
from numpy.random import default_rng

from sleplet.meshes.classes.mesh_slepian import MeshSlepian
from sleplet.slepian.slepian_functions import SlepianFunctions
from sleplet.utils.harmonic_methods import mesh_forward
from sleplet.utils.logger import logger
from sleplet.utils.slepian_methods import (
    compute_mesh_s_p_pixel,
    compute_s_p_omega,
    slepian_forward,
    slepian_inverse,
    slepian_mesh_forward,
    slepian_mesh_inverse,
)
from sleplet.utils.vars import RANDOM_SEED, SAMPLING_SCHEME


def _signal_power(signal: np.ndarray) -> float:
    """
    computes the power of the signal
    """
    return (np.abs(signal) ** 2).sum()


def compute_snr(signal: np.ndarray, noise: np.ndarray, signal_type: str) -> float:
    """
    computes the signal to noise ratio
    """
    snr = 10 * np.log10(_signal_power(signal) / _signal_power(noise))
    logger.info(f"{signal_type} SNR: {snr:.2f}")
    return snr


def compute_sigma_noise(
    signal: np.ndarray,
    snr_in: int,
    *,
    denominator: Optional[int] = None,
) -> float:
    """
    compute the std dev of the noise
    """
    if denominator is None:
        denominator = signal.shape[0]
    return np.sqrt(10 ** (-snr_in / 10) * _signal_power(signal) / denominator)


def create_noise(L: int, signal: np.ndarray, snr_in: int) -> np.ndarray:
    """
    computes Gaussian white noise
    """
    # set random seed
    rng = default_rng(RANDOM_SEED)

    # initialise
    nlm = np.zeros(L**2, dtype=np.complex_)

    # std dev of the noise
    sigma_noise = compute_sigma_noise(signal, snr_in)

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
        slepian_inverse(slepian_signal, L, slepian), L, Method=SAMPLING_SCHEME
    )
    nlm = create_noise(L, flm, snr_in)
    return slepian_forward(L, slepian, flm=nlm)


def _perform_hard_thresholding(
    f: np.ndarray, sigma_j: float | np.ndarray, n_sigma: int
) -> np.ndarray:
    """
    set pixels in real space to zero if the magnitude is less than the threshold
    """
    threshold = n_sigma * sigma_j
    return np.where(np.abs(f) < threshold, 0, f)


def harmonic_hard_thresholding(
    L: int, wav_coeffs: np.ndarray, sigma_j: np.ndarray, n_sigma: int
) -> np.ndarray:
    """
    perform thresholding in harmonic space
    """
    logger.info("begin harmonic hard thresholding")
    for j, coefficient in enumerate(wav_coeffs[1:]):
        logger.info(f"start Psi^{j + 1}/{len(wav_coeffs)-1}")
        f = ssht.inverse(coefficient, L, Method=SAMPLING_SCHEME)
        f_thresholded = _perform_hard_thresholding(f, sigma_j[j], n_sigma)
        wav_coeffs[j + 1] = ssht.forward(f_thresholded, L, Method=SAMPLING_SCHEME)
    return wav_coeffs


def slepian_wavelet_hard_thresholding(
    L: int,
    wav_coeffs: np.ndarray,
    sigma_j: np.ndarray,
    n_sigma: int,
    slepian: SlepianFunctions,
) -> np.ndarray:
    """
    perform thresholding in Slepian wavelet space
    """
    logger.info("begin Slepian hard thresholding")
    for j, coefficient in enumerate(wav_coeffs):
        logger.info(f"start Psi^{j + 1}/{len(wav_coeffs)}")
        f = slepian_inverse(coefficient, L, slepian)
        f_thresholded = _perform_hard_thresholding(f, sigma_j[j], n_sigma)
        wav_coeffs[j] = slepian_forward(L, slepian, f=f_thresholded)
    return wav_coeffs


def slepian_function_hard_thresholding(
    L: int,
    coefficients: np.ndarray,
    sigma: float,
    n_sigma: int,
    slepian: SlepianFunctions,
) -> np.ndarray:
    """
    perform thresholding in Slepian function space
    """
    logger.info("begin Slepian hard thresholding")
    f = slepian_inverse(coefficients, L, slepian)
    f_thresholded = _perform_hard_thresholding(f, sigma, n_sigma)
    return slepian_forward(L, slepian, f=f_thresholded)


def compute_sigma_j(signal: np.ndarray, psi_j: np.ndarray, snr_in: int) -> np.ndarray:
    """
    compute sigma_j for wavelets used in denoising the signal
    """
    sigma_noise = compute_sigma_noise(signal, snr_in)
    wavelet_power = (np.abs(psi_j) ** 2).sum(axis=1)
    return sigma_noise * np.sqrt(wavelet_power)


def compute_slepian_sigma_j(
    L: int,
    signal: np.ndarray,
    psi_j: np.ndarray,
    snr_in: int,
    slepian: SlepianFunctions,
) -> np.ndarray:
    """
    compute sigma_j for wavelets used in denoising the signal
    """
    sigma_noise = compute_sigma_noise(signal, snr_in, denominator=L**2)
    s_p = compute_s_p_omega(L, slepian)
    psi_j_reshape = psi_j[:, : slepian.N, np.newaxis, np.newaxis]
    wavelet_power = (np.abs(psi_j_reshape) ** 2 * np.abs(s_p) ** 2).sum(axis=1)
    return sigma_noise * np.sqrt(wavelet_power)


def create_mesh_noise(u_i: np.ndarray, snr_in: int) -> np.ndarray:
    """
    computes Gaussian white noise
    """
    # set random seed
    rng = default_rng(RANDOM_SEED)

    # initialise
    n_i = np.zeros(u_i.shape[0])

    # std dev of the noise
    sigma_noise = compute_sigma_noise(u_i, snr_in)

    # compute noise
    for i in range(u_i.shape[0]):
        n_i[i] = sigma_noise * rng.standard_normal()
    return n_i


def create_slepian_mesh_noise(
    mesh_slepian: MeshSlepian,
    slepian_signal: np.ndarray,
    snr_in: int,
) -> np.ndarray:
    """
    computes Gaussian white noise in Slepian space
    """
    u_i = mesh_forward(
        mesh_slepian.mesh,
        slepian_mesh_inverse(
            mesh_slepian,
            slepian_signal,
        ),
    )
    n_i = create_mesh_noise(u_i, snr_in)
    return slepian_mesh_forward(
        mesh_slepian,
        u_i=n_i,
    )


def compute_slepian_mesh_sigma_j(
    mesh_slepian: MeshSlepian,
    signal: np.ndarray,
    psi_j: np.ndarray,
    snr_in: int,
) -> np.ndarray:
    """
    compute sigma_j for wavelets used in denoising the signal
    """
    sigma_noise = compute_sigma_noise(
        signal, snr_in, denominator=mesh_slepian.slepian_eigenvalues.shape[0]
    )
    s_p = compute_mesh_s_p_pixel(mesh_slepian)
    psi_j_reshape = psi_j[:, : mesh_slepian.N, np.newaxis]
    wavelet_power = (np.abs(psi_j_reshape) ** 2 * np.abs(s_p) ** 2).sum(axis=1)
    return sigma_noise * np.sqrt(wavelet_power)


def slepian_mesh_hard_thresholding(
    mesh_slepian: MeshSlepian,
    wav_coeffs: np.ndarray,
    sigma_j: np.ndarray,
    n_sigma: int,
) -> np.ndarray:
    """
    perform thresholding in Slepian space
    """
    logger.info("begin Slepian mesh hard thresholding")
    for j, coefficient in enumerate(wav_coeffs):
        logger.info(f"start Psi^{j + 1}/{len(wav_coeffs)}")
        f = slepian_mesh_inverse(mesh_slepian, coefficient)
        f_thresholded = _perform_hard_thresholding(f, sigma_j[j], n_sigma)
        wav_coeffs[j] = slepian_mesh_forward(mesh_slepian, u=f_thresholded)
    return wav_coeffs
