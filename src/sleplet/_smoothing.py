import logging

import numpy as np
import numpy.typing as npt

import pyssht as ssht
import s2fft

_logger = logging.getLogger(__name__)


def apply_gaussian_smoothing(
    flm: npt.NDArray[np.complex_],
    L: int,
    smoothing_factor: int,
) -> npt.NDArray[np.complex_]:
    """
    Applies Gaussian smoothing to the given signal.

    s_lm = exp(-ell^2 sigma^2)
    s(omega) = exp(-theta^2 / sigma^2)
    FWHM = 2 * sqrt(ln2) * sigma
    """
    sigma = np.pi / (smoothing_factor * L)
    fwhm = 2 * np.sqrt(np.log(2)) * sigma
    _logger.info(f"FWHM = {np.rad2deg(fwhm):.2f}degrees")
    return s2fft.samples.flm_1d_to_2d(
        ssht.gaussian_smoothing(s2fft.samples.flm_2d_to_1d(flm, L), L, sigma),
        L,
    )
