import numpy as np
from numpy import typing as npt

from sleplet.functions import SlepianAfrica, SlepianSouthAmerica, SlepianWavelets
from sleplet.noise import (
    _compute_slepian_sigma_j,
    compute_snr,
    slepian_wavelet_hard_thresholding,
)
from sleplet.slepian_methods import slepian_inverse
from sleplet.wavelet_methods import slepian_wavelet_forward, slepian_wavelet_inverse


def denoising_slepian_wavelet(
    signal: SlepianAfrica | SlepianSouthAmerica,
    noised_signal: SlepianAfrica | SlepianSouthAmerica,
    slepian_wavelets: SlepianWavelets,
    snr_in: float,
    n_sigma: int,
) -> npt.NDArray[np.complex_]:
    """Denoising demo using Slepian wavelets."""
    # compute wavelet coefficients
    w = slepian_wavelet_forward(
        noised_signal.coefficients,
        slepian_wavelets.wavelets,
        slepian_wavelets.slepian.N,
    )

    # compute wavelet noise
    sigma_j = _compute_slepian_sigma_j(
        signal.L,
        signal.coefficients,
        slepian_wavelets.wavelets,
        snr_in,
        slepian_wavelets.slepian,
    )

    # hard thresholding
    w_denoised = slepian_wavelet_hard_thresholding(
        signal.L,
        w,
        sigma_j,
        n_sigma,
        slepian_wavelets.slepian,
    )

    # wavelet synthesis
    f_p = slepian_wavelet_inverse(
        w_denoised,
        slepian_wavelets.wavelets,
        slepian_wavelets.slepian.N,
    )

    # compute SNR
    compute_snr(signal.coefficients, f_p - signal.coefficients, "Slepian")

    return slepian_inverse(f_p, signal.L, slepian_wavelets.slepian)
