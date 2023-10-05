import numpy as np
import numpy.typing as npt

import s2fft

import sleplet

EXECUTION_MODE = "jax"
SAMPLING_SCHEME = "mwss"


def denoising_axisym(  # noqa: PLR0913
    signal: sleplet.functions.Africa
    | sleplet.functions.Earth
    | sleplet.functions.SouthAmerica,
    noised_signal: sleplet.functions.Africa
    | sleplet.functions.Earth
    | sleplet.functions.SouthAmerica,
    axisymmetric_wavelets: sleplet.functions.AxisymmetricWavelets,
    snr_in: float,
    n_sigma: int,
    *,
    rotate_to_south_america: bool = False,
) -> tuple[npt.NDArray[np.complex_], float | None, float]:
    """Reproduce the denoising demo from S2LET paper."""
    # compute wavelet coefficients
    w = sleplet.wavelet_methods.axisymmetric_wavelet_forward(
        signal.L,
        noised_signal.coefficients,
        axisymmetric_wavelets.wavelets,
    )

    # compute wavelet noise
    sigma_j = sleplet.noise._compute_sigma_j(
        signal.coefficients,
        axisymmetric_wavelets.wavelets[1:],
        snr_in,
    )

    # hard thresholding
    w_denoised = sleplet.noise.harmonic_hard_thresholding(signal.L, w, sigma_j, n_sigma)

    # wavelet synthesis
    flm = sleplet.wavelet_methods.axisymmetric_wavelet_inverse(
        signal.L,
        w_denoised,
        axisymmetric_wavelets.wavelets,
    )

    # compute SNR
    denoised_snr = sleplet.noise.compute_snr(
        signal.coefficients,
        flm - signal.coefficients,
        "Harmonic",
    )

    # rotate to South America
    flm = (
        sleplet.harmonic_methods.rotate_earth_to_south_america(flm, signal.L)
        if rotate_to_south_america
        else flm
    )

    f = s2fft.inverse(flm, signal.L, method=EXECUTION_MODE, sampling=SAMPLING_SCHEME)
    return f, noised_signal.snr, denoised_snr
