import numpy as np
import pyssht as ssht
from numpy import typing as npt

from sleplet._noise import (
    compute_sigma_j,
    compute_sigma_noise,
    compute_slepian_mesh_sigma_j,
    compute_slepian_sigma_j,
    compute_snr,
    harmonic_hard_thresholding,
    slepian_function_hard_thresholding,
    slepian_mesh_hard_thresholding,
    slepian_wavelet_hard_thresholding,
)
from sleplet._vars import SAMPLING_SCHEME
from sleplet.functions.coefficients import Coefficients
from sleplet.functions.f_p import F_P
from sleplet.functions.flm.axisymmetric_wavelets import AxisymmetricWavelets
from sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from sleplet.harmonic_methods import rotate_earth_to_south_america
from sleplet.meshes.mesh_coefficients import MeshCoefficients
from sleplet.meshes.slepian_coefficients.mesh_slepian_wavelets import (
    MeshSlepianWavelets,
)
from sleplet.slepian_methods import slepian_inverse, slepian_mesh_inverse
from sleplet.wavelet_methods import (
    axisymmetric_wavelet_forward,
    axisymmetric_wavelet_inverse,
    slepian_wavelet_forward,
    slepian_wavelet_inverse,
)


def denoising_axisym(
    signal: Coefficients,
    noised_signal: Coefficients,
    axisymmetric_wavelets: AxisymmetricWavelets,
    snr_in: float,
    n_sigma: int,
    *,
    rotate_to_south_america: bool = False,
) -> tuple[npt.NDArray[np.complex_], float | None, float]:
    """
    reproduce the denoising demo from s2let paper
    """
    # compute wavelet coefficients
    w = axisymmetric_wavelet_forward(
        signal.L, noised_signal.coefficients, axisymmetric_wavelets.wavelets
    )

    # compute wavelet noise
    sigma_j = compute_sigma_j(
        signal.coefficients, axisymmetric_wavelets.wavelets[1:], snr_in
    )

    # hard thresholding
    w_denoised = harmonic_hard_thresholding(signal.L, w, sigma_j, n_sigma)

    # wavelet synthesis
    flm = axisymmetric_wavelet_inverse(
        signal.L, w_denoised, axisymmetric_wavelets.wavelets
    )

    # compute SNR
    denoised_snr = compute_snr(
        signal.coefficients, flm - signal.coefficients, "Harmonic"
    )

    # rotate to South America
    flm = (
        rotate_earth_to_south_america(flm, signal.L) if rotate_to_south_america else flm
    )

    f = ssht.inverse(flm, signal.L, Method=SAMPLING_SCHEME)
    return f, noised_signal.snr, denoised_snr


def denoising_slepian_wavelet(
    signal: Coefficients,
    noised_signal: Coefficients,
    slepian_wavelets: SlepianWavelets,
    snr_in: float,
    n_sigma: int,
) -> npt.NDArray[np.complex_]:
    """
    denoising demo using Slepian wavelets
    """
    # compute wavelet coefficients
    w = slepian_wavelet_forward(
        noised_signal.coefficients,
        slepian_wavelets.wavelets,
        slepian_wavelets.slepian.N,
    )

    # compute wavelet noise
    sigma_j = compute_slepian_sigma_j(
        signal.L,
        signal.coefficients,
        slepian_wavelets.wavelets,
        snr_in,
        slepian_wavelets.slepian,
    )

    # hard thresholding
    w_denoised = slepian_wavelet_hard_thresholding(
        signal.L, w, sigma_j, n_sigma, slepian_wavelets.slepian
    )

    # wavelet synthesis
    f_p = slepian_wavelet_inverse(
        w_denoised, slepian_wavelets.wavelets, slepian_wavelets.slepian.N
    )

    # compute SNR
    compute_snr(signal.coefficients, f_p - signal.coefficients, "Slepian")

    return slepian_inverse(f_p, signal.L, slepian_wavelets.slepian)


def denoising_slepian_function(
    signal: F_P,
    noised_signal: F_P,
    snr_in: float,
    n_sigma: int,
) -> npt.NDArray[np.complex_]:
    """
    denoising demo using Slepian function
    """
    # compute Slepian noise
    sigma_noise = compute_sigma_noise(
        signal.coefficients, snr_in, denominator=signal.L**2
    )

    # hard thresholding
    f_p = slepian_function_hard_thresholding(
        signal.L, noised_signal.coefficients, sigma_noise, n_sigma, signal.slepian
    )

    # compute SNR
    compute_snr(signal.coefficients, f_p - signal.coefficients, "Slepian")

    return slepian_inverse(f_p, signal.L, signal.slepian)


def denoising_mesh_slepian(
    signal: MeshCoefficients,
    noised_signal: MeshCoefficients,
    mesh_slepian_wavelets: MeshSlepianWavelets,
    snr_in: float,
    n_sigma: int,
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    denoising demo using Slepian wavelets
    """
    # compute wavelet coefficients
    w = slepian_wavelet_forward(
        noised_signal.coefficients,
        mesh_slepian_wavelets.wavelets,
        mesh_slepian_wavelets.mesh_slepian.N,
    )

    # compute wavelet noise
    sigma_j = compute_slepian_mesh_sigma_j(
        mesh_slepian_wavelets.mesh_slepian,
        signal.coefficients,
        mesh_slepian_wavelets.wavelets,
        snr_in,
    )

    # hard thresholding
    w_denoised = slepian_mesh_hard_thresholding(
        mesh_slepian_wavelets.mesh_slepian, w, sigma_j, n_sigma
    )

    # wavelet synthesis
    f_p = slepian_wavelet_inverse(
        w_denoised, mesh_slepian_wavelets.wavelets, mesh_slepian_wavelets.mesh_slepian.N
    )

    # compute SNR
    compute_snr(signal.coefficients, f_p - signal.coefficients, "Slepian")

    return slepian_mesh_inverse(mesh_slepian_wavelets.mesh_slepian, f_p)
