import numpy as np
import pyssht as ssht


def apply_gaussian_smoothing(
    flm: np.ndarray, L: int, smoothing_factor: int
) -> np.ndarray:
    """
    applies Gaussian smoothing to the given signal
    """
    sigma = np.pi / (smoothing_factor * L)
    return ssht.gaussian_smoothing(flm, L, sigma)
