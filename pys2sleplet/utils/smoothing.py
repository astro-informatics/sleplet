import numpy as np
import pyssht as ssht


def apply_gaussian_smoothing(flm: np.ndarray, L: int) -> np.ndarray:
    """
    applies Gaussian smoothing to the given signal
    """
    sigma = np.pi / L
    return ssht.gaussian_smoothing(flm, L, sigma)
