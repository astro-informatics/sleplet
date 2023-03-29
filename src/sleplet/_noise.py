import numpy as np
import pyssht as ssht
from numpy import typing as npt
from numpy.random import default_rng

from sleplet import logger
from sleplet._vars import RANDOM_SEED, SAMPLING_SCHEME
from sleplet.harmonic_methods import mesh_forward
from sleplet.meshes.mesh_slepian import MeshSlepian
from sleplet.slepian.slepian_functions import SlepianFunctions
from sleplet.slepian_methods import (
    _compute_mesh_s_p_pixel,
    _compute_s_p_omega,
    slepian_forward,
    slepian_inverse,
    slepian_mesh_forward,
    slepian_mesh_inverse,
)


def _signal_power(signal: npt.NDArray[np.complex_ | np.float_]) -> float:
    """
    computes the power of the signal
    """
    return (np.abs(signal) ** 2).sum()


def compute_snr(
    signal: npt.NDArray[np.complex_ | np.float_],
    noise: npt.NDArray[np.complex_ | np.float_],
    signal_type: str,
) -> float:
    """
    computes the signal to noise ratio
    """
    snr = 10 * np.log10(_signal_power(signal) / _signal_power(noise))
    logger.info(f"{signal_type} SNR: {snr:.2f}")
    return snr


def compute_sigma_noise(
    signal: npt.NDArray[np.complex_ | np.float_],
    snr_in: float,
    *,
    denominator: int | None = None,
) -> float:
    """
    compute the std dev of the noise
    """
    if denominator is None:
        denominator = signal.shape[0]
    return np.sqrt(10 ** (-snr_in / 10) * _signal_power(signal) / denominator)


def create_noise(
    L: int, signal: npt.NDArray[np.complex_ | np.float_], snr_in: float
) -> npt.NDArray[np.complex_]:
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
    L: int,
    slepian_signal: npt.NDArray[np.complex_ | np.float_],
    slepian: SlepianFunctions,
    snr_in: float,
) -> npt.NDArray[np.complex_]:
    """
    computes Gaussian white noise in Slepian space
    """
    flm = ssht.forward(
        slepian_inverse(slepian_signal, L, slepian), L, Method=SAMPLING_SCHEME
    )
    nlm = create_noise(L, flm, snr_in)
    return slepian_forward(L, slepian, flm=nlm)


def _perform_hard_thresholding(
    f: npt.NDArray[np.complex_ | np.float_],
    sigma_j: float | npt.NDArray[np.float_],
    n_sigma: int,
) -> npt.NDArray[np.complex_]:
    """
    set pixels in real space to zero if the magnitude is less than the threshold
    """
    threshold = n_sigma * sigma_j
    return np.where(np.abs(f) < threshold, 0, f)


def harmonic_hard_thresholding(
    L: int,
    wav_coeffs: npt.NDArray[np.complex_],
    sigma_j: npt.NDArray[np.float_],
    n_sigma: int,
) -> npt.NDArray[np.complex_]:
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
    wav_coeffs: npt.NDArray[np.complex_ | np.float_],
    sigma_j: npt.NDArray[np.float_],
    n_sigma: int,
    slepian: SlepianFunctions,
) -> npt.NDArray[np.complex_ | np.float_]:
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
    coefficients: npt.NDArray[np.complex_ | np.float_],
    sigma: float,
    n_sigma: int,
    slepian: SlepianFunctions,
) -> npt.NDArray[np.complex_]:
    """
    perform thresholding in Slepian function space
    """
    logger.info("begin Slepian hard thresholding")
    f = slepian_inverse(coefficients, L, slepian)
    f_thresholded = _perform_hard_thresholding(f, sigma, n_sigma)
    return slepian_forward(L, slepian, f=f_thresholded)


def compute_sigma_j(
    signal: npt.NDArray[np.complex_ | np.float_],
    psi_j: npt.NDArray[np.complex_],
    snr_in: float,
) -> npt.NDArray[np.float_]:
    """
    compute sigma_j for wavelets used in denoising the signal
    """
    sigma_noise = compute_sigma_noise(signal, snr_in)
    wavelet_power = (np.abs(psi_j) ** 2).sum(axis=1)
    return sigma_noise * np.sqrt(wavelet_power)


def compute_slepian_sigma_j(
    L: int,
    signal: npt.NDArray[np.complex_ | np.float_],
    psi_j: npt.NDArray[np.float_],
    snr_in: float,
    slepian: SlepianFunctions,
) -> npt.NDArray[np.float_]:
    """
    compute sigma_j for wavelets used in denoising the signal
    """
    sigma_noise = compute_sigma_noise(signal, snr_in, denominator=L**2)
    s_p = _compute_s_p_omega(L, slepian)
    psi_j_reshape = psi_j[:, : slepian.N, np.newaxis, np.newaxis]
    wavelet_power = (np.abs(psi_j_reshape) ** 2 * np.abs(s_p) ** 2).sum(axis=1)
    return sigma_noise * np.sqrt(wavelet_power)


def create_mesh_noise(
    u_i: npt.NDArray[np.complex_ | np.float_], snr_in: float
) -> npt.NDArray[np.float_]:
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
    slepian_signal: npt.NDArray[np.complex_ | np.float_],
    snr_in: float,
) -> npt.NDArray[np.float_]:
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
    signal: npt.NDArray[np.complex_ | np.float_],
    psi_j: npt.NDArray[np.float_],
    snr_in: float,
) -> npt.NDArray[np.float_]:
    """
    compute sigma_j for wavelets used in denoising the signal
    """
    sigma_noise = compute_sigma_noise(
        signal, snr_in, denominator=mesh_slepian.slepian_eigenvalues.shape[0]
    )
    s_p = _compute_mesh_s_p_pixel(mesh_slepian)
    psi_j_reshape = psi_j[:, : mesh_slepian.N, np.newaxis]
    wavelet_power = (np.abs(psi_j_reshape) ** 2 * np.abs(s_p) ** 2).sum(axis=1)
    return sigma_noise * np.sqrt(wavelet_power)


def slepian_mesh_hard_thresholding(
    mesh_slepian: MeshSlepian,
    wav_coeffs: npt.NDArray[np.complex_ | np.float_],
    sigma_j: npt.NDArray[np.float_],
    n_sigma: int,
) -> npt.NDArray[np.complex_ | np.float_]:
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
