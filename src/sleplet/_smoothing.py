import logging

import numpy as np
import numpy.typing as npt

import pyssht as ssht

_logger = logging.getLogger(__name__)


def apply_gaussian_smoothing(
    flm: npt.NDArray[np.complex_],
    L: int,
    smoothing_factor: int,
) -> npt.NDArray[np.complex_]:
    """
    Apply Gaussian smoothing to the given signal.

    s_lm = exp(-ell^2 sigma^2)
    s(omega) = exp(-theta^2 / sigma^2)
    FWHM = 2 * sqrt(ln2) * sigma
    """
    sigma = np.pi / (smoothing_factor * L)
    fwhm = 2 * np.sqrt(np.log(2)) * sigma
    msg = f"FWHM = {np.rad2deg(fwhm):.2f}degrees"
    _logger.info(msg)
    return ssht.gaussian_smoothing(flm, L, sigma)
