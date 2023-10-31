"""Methods to handle noise in Fourier or wavelet space."""
import logging

import numpy as np
import numpy.typing as npt

import pyssht as ssht

import sleplet._vars
import sleplet.harmonic_methods
import sleplet.meshes.mesh_slepian
import sleplet.slepian_methods
from sleplet.slepian.slepian_functions import SlepianFunctions

_logger = logging.getLogger(__name__)


def _signal_power(signal: npt.NDArray[np.complex_ | np.float_]) -> float:
    """Compute the power of the signal."""
    return (np.abs(signal) ** 2).sum()


def compute_snr(
    signal: npt.NDArray[np.complex_ | np.float_],
    noise: npt.NDArray[np.complex_ | np.float_],
    signal_type: str,
) -> float:
    """
    Compute the signal to noise ratio.

    Args:
        signal: The unnoised signal.
        noise: The noised signal.
        signal_type: Specifier to improve logging.

    Returns:
        The signal-to-noise value of the noised signal.
    """
    snr = 10 * np.log10(_signal_power(signal) / _signal_power(noise))
    msg = f"{signal_type} SNR: {snr:.2f}"
    _logger.info(msg)
    return snr


def compute_sigma_noise(
    signal: npt.NDArray[np.complex_ | np.float_],
    snr_in: float,
    *,
    denominator: int | None = None,
) -> float:
    """
    Compute the standard deviation of the noise.

    Args:
        signal: The noised signal.
        snr_in: The parameter controlling the signal-to-noise.
        denominator: Coefficients to use in computing the noise, defaults to all.

    Returns:
        The standard deviation of the noise.
    """
    if denominator is None:
        denominator = signal.shape[0]
    return np.sqrt(10 ** (-snr_in / 10) * _signal_power(signal) / denominator)


def _create_noise(
    L: int,
    signal: npt.NDArray[np.complex_ | np.float_],
    snr_in: float,
) -> npt.NDArray[np.complex_]:
    """Compute Gaussian white noise."""
    # set random seed
    rng = np.random.default_rng(sleplet._vars.RANDOM_SEED)

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


def _create_slepian_noise(
    L: int,
    slepian_signal: npt.NDArray[np.complex_ | np.float_],
    slepian: SlepianFunctions,
    snr_in: float,
) -> npt.NDArray[np.complex_]:
    """Compute Gaussian white noise in Slepian space."""
    flm = ssht.forward(
        sleplet.slepian_methods.slepian_inverse(slepian_signal, L, slepian),
        L,
        Method=sleplet._vars.SAMPLING_SCHEME,
    )
    nlm = _create_noise(L, flm, snr_in)
    return sleplet.slepian_methods.slepian_forward(L, slepian, flm=nlm)


def _perform_hard_thresholding(
    f: npt.NDArray[np.complex_ | np.float_],
    sigma_j: float | npt.NDArray[np.float_],
    n_sigma: int,
) -> npt.NDArray[np.complex_]:
    """Set pixels in pixel space to zero if the magnitude is less than the threshold."""
    threshold = n_sigma * sigma_j
    return np.where(np.abs(f) < threshold, 0, f)


def harmonic_hard_thresholding(
    L: int,
    wav_coeffs: npt.NDArray[np.complex_],
    sigma_j: npt.NDArray[np.float_],
    n_sigma: int,
) -> npt.NDArray[np.complex_]:
    r"""
    Perform thresholding in harmonic space.

    Args:
        L: The spherical harmonic bandlimit.
        wav_coeffs: The harmonic wavelet coefficients.
        sigma_j: The wavelet standard deviation \(\sigma_{j}\).
        n_sigma: The number of \(\sigma_{j}\) to threshold.

    Returns:
        The thresholded wavelet coefficients.
    """
    _logger.info("begin harmonic hard thresholding")
    for j, coefficient in enumerate(wav_coeffs[1:]):
        msg = f"start Psi^{j + 1}/{len(wav_coeffs)-1}"
        _logger.info(msg)
        f = ssht.inverse(coefficient, L, Method=sleplet._vars.SAMPLING_SCHEME)
        f_thresholded = _perform_hard_thresholding(f, sigma_j[j], n_sigma)
        wav_coeffs[j + 1] = ssht.forward(
            f_thresholded,
            L,
            Method=sleplet._vars.SAMPLING_SCHEME,
        )
    return wav_coeffs


def slepian_wavelet_hard_thresholding(
    L: int,
    wav_coeffs: npt.NDArray[np.complex_ | np.float_],
    sigma_j: npt.NDArray[np.float_],
    n_sigma: int,
    slepian: SlepianFunctions,
) -> npt.NDArray[np.complex_ | np.float_]:
    r"""
    Perform thresholding in Slepian wavelet space.

    Args:
        L: The spherical harmonic bandlimit.
        wav_coeffs: The Slepian wavelet coefficients
        sigma_j: The wavelet standard deviation \(\sigma_{j}\).
        n_sigma: The number of \(\sigma_{j}\) to threshold.
        slepian: The given Slepian object.

    Returns:
        The hard thresholded Slepian wavelet coefficients.
    """
    _logger.info("begin Slepian hard thresholding")
    for j, coefficient in enumerate(wav_coeffs):
        msg = f"start Psi^{j + 1}/{len(wav_coeffs)}"
        _logger.info(msg)
        f = sleplet.slepian_methods.slepian_inverse(coefficient, L, slepian)
        f_thresholded = _perform_hard_thresholding(f, sigma_j[j], n_sigma)
        wav_coeffs[j] = sleplet.slepian_methods.slepian_forward(
            L,
            slepian,
            f=f_thresholded,
        )
    return wav_coeffs


def slepian_function_hard_thresholding(
    L: int,
    coefficients: npt.NDArray[np.complex_ | np.float_],
    sigma: float,
    n_sigma: int,
    slepian: SlepianFunctions,
) -> npt.NDArray[np.complex_]:
    r"""
    Perform thresholding in Slepian space.

    Args:
        L: The spherical harmonic bandlimit.
        coefficients: The Slepian coefficients.
        sigma: The standard deviation of the noise \(\sigma\).
        n_sigma: The number of \(\sigma\) to threshold.
        slepian: The given Slepian object.

    Returns:
        The thresholded Slepian coefficients.
    """
    _logger.info("begin Slepian hard thresholding")
    f = sleplet.slepian_methods.slepian_inverse(coefficients, L, slepian)
    f_thresholded = _perform_hard_thresholding(f, sigma, n_sigma)
    return sleplet.slepian_methods.slepian_forward(L, slepian, f=f_thresholded)


def _compute_sigma_j(
    signal: npt.NDArray[np.complex_ | np.float_],
    psi_j: npt.NDArray[np.complex_],
    snr_in: float,
) -> npt.NDArray[np.float_]:
    """Compute sigma_j for wavelets used in denoising the signal."""
    sigma_noise = compute_sigma_noise(signal, snr_in)
    wavelet_power = (np.abs(psi_j) ** 2).sum(axis=1)
    return sigma_noise * np.sqrt(wavelet_power)


def _compute_slepian_sigma_j(
    L: int,
    signal: npt.NDArray[np.complex_ | np.float_],
    psi_j: npt.NDArray[np.float_],
    snr_in: float,
    slepian: SlepianFunctions,
) -> npt.NDArray[np.float_]:
    """Compute sigma_j for wavelets used in denoising the signal."""
    sigma_noise = compute_sigma_noise(signal, snr_in, denominator=L**2)
    s_p = sleplet.slepian_methods.compute_s_p_omega(L, slepian)
    psi_j_reshape = psi_j[:, : slepian.N, np.newaxis, np.newaxis]
    wavelet_power = (np.abs(psi_j_reshape) ** 2 * np.abs(s_p) ** 2).sum(axis=1)
    return sigma_noise * np.sqrt(wavelet_power)


def _create_mesh_noise(
    u_i: npt.NDArray[np.complex_ | np.float_],
    snr_in: float,
) -> npt.NDArray[np.float_]:
    """Compute Gaussian white noise."""
    # set random seed
    rng = np.random.default_rng(sleplet._vars.RANDOM_SEED)

    # initialise
    n_i = np.zeros(u_i.shape[0])

    # std dev of the noise
    sigma_noise = compute_sigma_noise(u_i, snr_in)

    # compute noise
    for i in range(u_i.shape[0]):
        n_i[i] = sigma_noise * rng.standard_normal()
    return n_i


def _create_slepian_mesh_noise(
    mesh_slepian: "sleplet.meshes.mesh_slepian.MeshSlepian",
    slepian_signal: npt.NDArray[np.complex_ | np.float_],
    snr_in: float,
) -> npt.NDArray[np.float_]:
    """Compute Gaussian white noise in Slepian space."""
    u_i = sleplet.harmonic_methods.mesh_forward(
        mesh_slepian.mesh,
        sleplet.slepian_methods.slepian_mesh_inverse(
            mesh_slepian,
            slepian_signal,
        ),
    )
    n_i = _create_mesh_noise(u_i, snr_in)
    return sleplet.slepian_methods.slepian_mesh_forward(
        mesh_slepian,
        u_i=n_i,
    )


def compute_slepian_mesh_sigma_j(
    mesh_slepian: "sleplet.meshes.mesh_slepian.MeshSlepian",
    signal: npt.NDArray[np.complex_ | np.float_],
    psi_j: npt.NDArray[np.float_],
    snr_in: float,
) -> npt.NDArray[np.float_]:
    r"""
    Compute \(\sigma_{j}\) for wavelets used in denoising the signal.

    Args:
        mesh_slepian: The Slepian mesh object containing the eigensolutions.
        signal: The noised signal.
        psi_j: The Slepian wavelet coefficients.
        snr_in: The parameter controlling the signal-to-noise ratio.

    Returns:
        The standard deviation of the noise.
    """
    sigma_noise = compute_sigma_noise(
        signal,
        snr_in,
        denominator=mesh_slepian.slepian_eigenvalues.shape[0],
    )
    s_p = sleplet.slepian_methods._compute_mesh_s_p_pixel(mesh_slepian)
    psi_j_reshape = psi_j[:, : mesh_slepian.N, np.newaxis]
    wavelet_power = (np.abs(psi_j_reshape) ** 2 * np.abs(s_p) ** 2).sum(axis=1)
    return sigma_noise * np.sqrt(wavelet_power)


def slepian_mesh_hard_thresholding(
    mesh_slepian: "sleplet.meshes.mesh_slepian.MeshSlepian",
    wav_coeffs: npt.NDArray[np.complex_ | np.float_],
    sigma_j: npt.NDArray[np.float_],
    n_sigma: int,
) -> npt.NDArray[np.complex_ | np.float_]:
    r"""
    Perform thresholding in Slepian space of the mesh.

    Args:
        mesh_slepian: The Slepian mesh object containing the eigensolutions.
        wav_coeffs: The Slepian wavelet coefficients of the mesh.
        sigma_j: The wavelet standard deviation \(\sigma_{j}\).
        n_sigma: The number of \(\sigma\) to threshold.

    Returns:
        The thresholded wavelet coefficients of the mesh.
    """
    _logger.info("begin Slepian mesh hard thresholding")
    for j, coefficient in enumerate(wav_coeffs):
        msg = f"start Psi^{j + 1}/{len(wav_coeffs)}"
        _logger.info(msg)
        f = sleplet.slepian_methods.slepian_mesh_inverse(mesh_slepian, coefficient)
        f_thresholded = _perform_hard_thresholding(f, sigma_j[j], n_sigma)
        wav_coeffs[j] = sleplet.slepian_methods.slepian_mesh_forward(
            mesh_slepian,
            u=f_thresholded,
        )
    return wav_coeffs
