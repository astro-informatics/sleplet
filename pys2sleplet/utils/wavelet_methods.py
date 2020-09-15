import numpy as np

from pys2sleplet.flm.functions import Functions


def wavelet_forward(f: Functions, wavelets: np.ndarray) -> np.ndarray:
    """
    computes the coefficient of the given tiling function in harmonic space
    """
    return f.convolve(wavelets, f.multipole)


def wavelet_inverse(wav_coeffs: np.ndarray, wavelets: np.ndarray) -> np.ndarray:
    """
    computes the inverse wavelet transform
    """
    return (wav_coeffs.conj() * wavelets).sum(axis=0)
