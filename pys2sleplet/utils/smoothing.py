import numpy as np
import pyssht as ssht

from pys2sleplet.utils.logger import logger


def apply_gaussian_smoothing(
    flm: np.ndarray, L: int, smoothing_factor: int
) -> np.ndarray:
    """
    applies Gaussian smoothing to the given signal

    s_lm = exp(-ell^2 sigma^2)
    s(omega) = exp(-theta^2 / sigma^2)
    FWHM = 2 * sqrt(ln2) * sigma
    """
    sigma = np.pi / (smoothing_factor * L)
    fwhm = 2 * np.sqrt(np.log(2)) * sigma
    logger.info(f"FWHM = {np.rad2deg(fwhm):.2f}degrees")
    return ssht.gaussian_smoothing(flm, L, sigma)
