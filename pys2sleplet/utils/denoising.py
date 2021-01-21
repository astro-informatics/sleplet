from typing import Tuple

import numpy as np
import pyssht as ssht

from pys2sleplet.functions.coefficients import Coefficients
from pys2sleplet.functions.flm.axisymmetric_wavelets import AxisymmetricWavelets
from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from pys2sleplet.utils.noise import (
    compute_sigma_j,
    compute_slepian_sigma_j,
    compute_snr,
    harmonic_hard_thresholding,
    slepian_hard_thresholding,
)
from pys2sleplet.utils.plot_methods import rotate_earth_to_south_america
from pys2sleplet.utils.slepian_methods import slepian_inverse
from pys2sleplet.utils.vars import SAMPLING_SCHEME
from pys2sleplet.utils.wavelet_methods import (
    axisymmetric_wavelet_forward,
    axisymmetric_wavelet_inverse,
    slepian_wavelet_forward,
    slepian_wavelet_inverse,
)


def denoising_axisym(
    signal: Coefficients,
    noised_signal: Coefficients,
    axisymmetric_wavelets: AxisymmetricWavelets,
    snr_in: int,
    n_sigma: int,
) -> Tuple[np.ndarray, float, float]:
    """
    reproduce the denoising demo from s2let paper
    """
    # compute wavelet coefficients
    w = axisymmetric_wavelet_forward(
        signal.L, noised_signal.coefficients, axisymmetric_wavelets.wavelets
    )

    # compute wavelet noise
    sigma_j = compute_sigma_j(
        signal.L, signal.coefficients, axisymmetric_wavelets.wavelets[1:], snr_in
    )

    # hard thresholding
    w_denoised = harmonic_hard_thresholding(signal.L, w, sigma_j, n_sigma)

    # wavelet synthesis
    flm = axisymmetric_wavelet_inverse(
        signal.L, w_denoised, axisymmetric_wavelets.wavelets
    )

    # rotate to South America
    flm_rot = rotate_earth_to_south_america(flm, signal.L)

    # compute SNR
    denoised_snr = compute_snr(signal.L, signal.coefficients, flm - signal.coefficients)

    f = ssht.inverse(flm_rot, signal.L, Method=SAMPLING_SCHEME)
    return f, noised_signal.snr, denoised_snr


def denoising_slepian(
    signal: Coefficients,
    noised_signal: Coefficients,
    slepian_wavelets: SlepianWavelets,
    snr_in: int,
    n_sigma: int,
) -> np.ndarray:
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
    w_denoised = slepian_hard_thresholding(
        signal.L, w, sigma_j, n_sigma, slepian_wavelets.slepian
    )

    # wavelet synthesis
    f_p = slepian_wavelet_inverse(
        w_denoised, slepian_wavelets.wavelets, slepian_wavelets.slepian.N
    )

    # compute SNR
    compute_snr(signal.L, signal.coefficients, f_p - signal.coefficients)

    return slepian_inverse(f_p, signal.L, slepian_wavelets.slepian)
