import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.logger import logger


def wavelet_forward(f: Functions, wavelets: np.ndarray) -> np.ndarray:
    """
    computes the coefficient of the given tiling function in harmonic space
    """
    if wavelets.shape[1] != f.L ** 2:
        logger.error(f"tiling should have length {f.L ** 2}")
    return f.convolve(wavelets, f.multipole)


def wavelet_inverse(f: Functions, wavelets: np.ndarray) -> np.ndarray:
    """
    computes the inverse wavelet transform for the given filing function
    """
    if wavelets.shape[1] != f.L ** 2:
        logger.error(f"wavelets should have length {f.L ** 2}")
    w_coefficient = wavelet_forward(f, wavelets)
    return (w_coefficient.conj() * wavelets).sum(axis=0)
