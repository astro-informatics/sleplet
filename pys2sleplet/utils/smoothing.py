import numpy as np
import pyssht as ssht


def apply_gaussian_smoothing(flm: np.ndarray, L: int, sigma: float) -> np.ndarray:
    """
    applies Gaussian smoothing to the given signal
    """
    for ell in range(L):
        gl0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * np.exp(
            -ell * (ell + 1) / (2 * sigma ** 2)
        )
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = flm[ind] * gl0.conj()
    return flm
