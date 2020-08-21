import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.logger import logger


def wavelet_forward(f: Functions, tiling: np.ndarray) -> np.ndarray:
    """
    computes the coefficient of the given tiling function in harmonic space
    """
    if len(tiling) != f.L ** 2:
        logger.error(f"tiling should have length {f.L ** 2}")
    return f.convolve(tiling, f.multipole)


def wavelet_inverse(f: Functions, tiling: np.ndarray) -> np.ndarray:
    """
    computes the inverse wavelet transform for the given filing function
    """
    if len(tiling) != f.L ** 2:
        logger.error(f"tiling should have length {f.L ** 2}")
    w_coefficient = wavelet_forward(f, tiling)
    return w_coefficient.conj() * tiling
