import numpy as np
import numpy.typing as npt

import sleplet


def denoising_slepian_wavelet(
    signal: sleplet.functions.SlepianAfrica | sleplet.functions.SlepianSouthAmerica,
    noised_signal: sleplet.functions.SlepianAfrica
    | sleplet.functions.SlepianSouthAmerica,
    slepian_wavelets: sleplet.functions.SlepianWavelets,
    snr_in: float,
    n_sigma: int,
) -> npt.NDArray[np.complex_]:
    """Denoising demo using Slepian wavelets."""
    # compute wavelet coefficients
    w = sleplet.wavelet_methods.slepian_wavelet_forward(
        noised_signal.coefficients,
        slepian_wavelets.wavelets,
        slepian_wavelets.slepian.N,
    )

    # compute wavelet noise
    sigma_j = sleplet.noise._compute_slepian_sigma_j(
        signal.L,
        signal.coefficients,
        slepian_wavelets.wavelets,
        snr_in,
        slepian_wavelets.slepian,
    )

    # hard thresholding
    w_denoised = sleplet.noise.slepian_wavelet_hard_thresholding(
        signal.L,
        w,
        sigma_j,
        n_sigma,
        slepian_wavelets.slepian,
    )

    # wavelet synthesis
    f_p = sleplet.wavelet_methods.slepian_wavelet_inverse(
        w_denoised,
        slepian_wavelets.wavelets,
        slepian_wavelets.slepian.N,
    )

    # compute SNR
    sleplet.noise.compute_snr(signal.coefficients, f_p - signal.coefficients, "Slepian")

    return sleplet.slepian_methods.slepian_inverse(
        f_p,
        signal.L,
        slepian_wavelets.slepian,
    )
