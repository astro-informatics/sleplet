import numpy as np
import pyssht as ssht

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


def axisymmetric_wavelet_forward(
    L: int, f: Functions, wavelets: np.ndarray
) -> np.ndarray:
    """
    computes the coefficients of the axisymmetric wavelets
    """
    w = np.zeros(wavelets.shape, dtype=np.complex128)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * wavelets[:, ind_m0].conj()
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            w[:, ind] = wav_0 * f.multipole[ind]
    return w


def axisymmetric_wavelet_inverse(
    L: int, wav_coeffs: np.ndarray, wavelets: np.ndarray
) -> np.ndarray:
    """
    computes the inverse axisymmetric wavelet transform
    """
    flm = np.zeros(L ** 2, dtype=np.complex128)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * wavelets[:, ind_m0]
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = (wav_coeffs[:, ind] * wav_0).sum()
    return flm
