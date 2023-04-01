import numpy as np
import pyssht as ssht
from numpy import typing as npt

from sleplet.functions import Africa, AxisymmetricWavelets, Earth, SouthAmerica
from sleplet.harmonic_methods import rotate_earth_to_south_america
from sleplet.noise import _compute_sigma_j, compute_snr, harmonic_hard_thresholding
from sleplet.wavelet_methods import (
    axisymmetric_wavelet_forward,
    axisymmetric_wavelet_inverse,
)

SAMPLING_SCHEME = "MWSS"


def denoising_axisym(
    signal: Africa | Earth | SouthAmerica,
    noised_signal: Africa | Earth | SouthAmerica,
    axisymmetric_wavelets: AxisymmetricWavelets,
    snr_in: float,
    n_sigma: int,
    *,
    rotate_to_south_america: bool = False,
) -> tuple[npt.NDArray[np.complex_], float | None, float]:
    """Reproduce the denoising demo from s2let paper."""
    # compute wavelet coefficients
    w = axisymmetric_wavelet_forward(
        signal.L,
        noised_signal.coefficients,
        axisymmetric_wavelets.wavelets,
    )

    # compute wavelet noise
    sigma_j = _compute_sigma_j(
        signal.coefficients,
        axisymmetric_wavelets.wavelets[1:],
        snr_in,
    )

    # hard thresholding
    w_denoised = harmonic_hard_thresholding(signal.L, w, sigma_j, n_sigma)

    # wavelet synthesis
    flm = axisymmetric_wavelet_inverse(
        signal.L,
        w_denoised,
        axisymmetric_wavelets.wavelets,
    )

    # compute SNR
    denoised_snr = compute_snr(
        signal.coefficients,
        flm - signal.coefficients,
        "Harmonic",
    )

    # rotate to South America
    flm = (
        rotate_earth_to_south_america(flm, signal.L) if rotate_to_south_america else flm
    )

    f = ssht.inverse(flm, signal.L, Method=SAMPLING_SCHEME)
    return f, noised_signal.snr, denoised_snr
