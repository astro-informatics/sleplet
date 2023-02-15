import numpy as np
import pyssht as ssht
from numpy import typing as npt
from pys2let import axisym_wav_l
from sleplet.slepian.slepian_functions import SlepianFunctions
from sleplet.utils.convolution_methods import sifting_convolution
from sleplet.utils.slepian_methods import compute_s_p_omega


def slepian_wavelet_forward(
    f_p: npt.NDArray[np.complex_ | np.float_],
    wavelets: npt.NDArray[np.float_],
    shannon: int,
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    computes the coefficients of the given tiling function in Slepian space
    """
    return find_non_zero_wavelet_coefficients(
        sifting_convolution(wavelets, f_p, shannon=shannon), axis=1
    )


def slepian_wavelet_inverse(
    wav_coeffs: npt.NDArray[np.complex_ | np.float_],
    wavelets: npt.NDArray[np.float_],
    shannon: int,
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    computes the inverse wavelet transform in Slepian space
    """
    # ensure wavelets are the same shape as the coefficients
    wavelets_shannon = wavelets[: len(wav_coeffs)]
    wavelet_reconstruction = sifting_convolution(
        wavelets_shannon, wav_coeffs.T, shannon=shannon
    )
    return wavelet_reconstruction.sum(axis=0)


def axisymmetric_wavelet_forward(
    L: int,
    flm: npt.NDArray[np.complex_ | np.float_],
    wavelets: npt.NDArray[np.complex_],
) -> npt.NDArray[np.complex_]:
    """
    computes the coefficients of the axisymmetric wavelets
    """
    w = np.zeros(wavelets.shape, dtype=np.complex_)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * wavelets[:, ind_m0].conj()
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            w[:, ind] = wav_0 * flm[ind]
    return w


def axisymmetric_wavelet_inverse(
    L: int, wav_coeffs: npt.NDArray[np.complex_], wavelets: npt.NDArray[np.complex_]
) -> npt.NDArray[np.complex_]:
    """
    computes the inverse axisymmetric wavelet transform
    """
    flm = np.zeros(L**2, dtype=np.complex_)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * wavelets[:, ind_m0]
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = (wav_coeffs[:, ind] * wav_0).sum()
    return flm


def compute_wavelet_covariance(
    L: int, wavelets: npt.NDArray[np.complex_], *, var_signal: float
) -> npt.NDArray[np.float_]:
    """
    computes the theoretical covariance of the wavelet coefficients
    """
    covar_theory = (np.abs(wavelets) ** 2).sum(axis=1)
    return covar_theory * var_signal


def compute_slepian_wavelet_covariance(
    L: int,
    wavelets: npt.NDArray[np.float_],
    slepian: SlepianFunctions,
    *,
    var_signal: float,
) -> npt.NDArray[np.float_]:
    """
    computes the theoretical covariance of the wavelet coefficients
    """
    s_p = compute_s_p_omega(L, slepian)
    wavelets_reshape = wavelets[:, : slepian.N, np.newaxis, np.newaxis]
    covar_theory = (np.abs(wavelets_reshape) ** 2 * np.abs(s_p) ** 2).sum(axis=1)
    return covar_theory * var_signal


def create_axisymmetric_wavelets(
    L: int, B: int, j_min: int
) -> npt.NDArray[np.complex_]:
    """
    computes the axisymmetric wavelets
    """
    kappas = create_kappas(L, B, j_min)
    wavelets = np.zeros((kappas.shape[0], L**2), dtype=np.complex_)
    for ell in range(L):
        factor = np.sqrt((2 * ell + 1) / (4 * np.pi))
        ind = ssht.elm2ind(ell, 0)
        wavelets[:, ind] = factor * kappas[:, ell]
    return wavelets


def create_kappas(xlim: int, B: int, j_min: int) -> npt.NDArray[np.float_]:
    """
    computes the Slepian wavelets
    """
    kappa0, kappa = axisym_wav_l(B, xlim, j_min)
    return np.concatenate((kappa0[np.newaxis], kappa.T))


def find_non_zero_wavelet_coefficients(
    wav_coeffs: npt.NDArray[np.complex_ | np.float_], *, axis: int | tuple[int, ...]
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    finds the coefficients within the shannon number to speed up computations
    """
    return wav_coeffs[wav_coeffs.any(axis=axis)]
