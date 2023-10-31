"""Methods to work with wavelet and wavelet coefficients."""
import numpy as np
import numpy.typing as npt

import pys2let
import pyssht as ssht

import sleplet._convolution_methods
import sleplet.slepian_methods


def slepian_wavelet_forward(
    f_p: npt.NDArray[np.complex_ | np.float_],
    wavelets: npt.NDArray[np.float_],
    shannon: int,
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    Compute the coefficients of the given tiling function in Slepian space.

    Args:
        f_p: The Slepian coefficients.
        wavelets: The Slepian wavelets.
        shannon: The Shannon number.

    Returns:
        The Slepian wavelets coefficients of the signal.
    """
    return find_non_zero_wavelet_coefficients(
        sleplet._convolution_methods.sifting_convolution(
            wavelets,
            f_p,
            shannon=shannon,
        ),
        axis=1,
    )


def slepian_wavelet_inverse(
    wav_coeffs: npt.NDArray[np.complex_ | np.float_],
    wavelets: npt.NDArray[np.float_],
    shannon: int,
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    Compute the inverse wavelet transform in Slepian space.

    Args:
        wav_coeffs: The Slepian wavelet coefficients.
        wavelets: The Slepian wavelets.
        shannon: The Shannon number.

    Returns:
        The coefficients of the signal in Slepian space.
    """
    # ensure wavelets are the same shape as the coefficients
    wavelets_shannon = wavelets[: len(wav_coeffs)]
    wavelet_reconstruction = sleplet._convolution_methods.sifting_convolution(
        wavelets_shannon,
        wav_coeffs,
        shannon=shannon,
    )
    return wavelet_reconstruction.sum(axis=0)


def axisymmetric_wavelet_forward(
    L: int,
    flm: npt.NDArray[np.complex_ | np.float_],
    wavelets: npt.NDArray[np.complex_],
) -> npt.NDArray[np.complex_]:
    """
    Compute the coefficients of the axisymmetric wavelets.

    Args:
        L: The spherical harmonic bandlimit.
        flm: The spherical harmonic coefficients.
        wavelets: Axisymmetric wavelets.

    Returns:
        Axisymmetric wavelets coefficients.
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
    L: int,
    wav_coeffs: npt.NDArray[np.complex_],
    wavelets: npt.NDArray[np.complex_],
) -> npt.NDArray[np.complex_]:
    """
    Compute the inverse axisymmetric wavelet transform.

    Args:
        L: The spherical harmonic bandlimit.
        wav_coeffs: Axisymmetric wavelet coefficients.
        wavelets: Axisymmetric wavelets.

    Returns:
        Spherical harmonic coefficients of the signal.
    """
    flm = np.zeros(L**2, dtype=np.complex_)
    for ell in range(L):
        ind_m0 = ssht.elm2ind(ell, 0)
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * wavelets[:, ind_m0]
        for m in range(-ell, ell + 1):
            ind = ssht.elm2ind(ell, m)
            flm[ind] = (wav_coeffs[:, ind] * wav_0).sum()
    return flm


def _create_axisymmetric_wavelets(
    L: int,
    B: int,
    j_min: int,
) -> npt.NDArray[np.complex_]:
    """Compute the axisymmetric wavelets."""
    kappas = create_kappas(L, B, j_min)
    wavelets = np.zeros((kappas.shape[0], L**2), dtype=np.complex_)
    for ell in range(L):
        factor = np.sqrt((2 * ell + 1) / (4 * np.pi))
        ind = ssht.elm2ind(ell, 0)
        wavelets[:, ind] = factor * kappas[:, ell]
    return wavelets


def create_kappas(xlim: int, B: int, j_min: int) -> npt.NDArray[np.float_]:
    r"""
    Compute the Slepian wavelets.

    Args:
        xlim: The x-axis value. \(L\) or \(L^2\) in the harmonic or Slepian case.
        B: The wavelet parameter. Represented as \(\lambda\) in the papers.
        j_min: The minimum wavelet scale. Represented as \(J_{0}\) in the papers.

    Returns:
        The Slepian wavelet generating functions.
    """
    kappa0, kappa = pys2let.axisym_wav_l(B, xlim, j_min)
    return np.concatenate((kappa0[np.newaxis], kappa.T))


def find_non_zero_wavelet_coefficients(
    wav_coeffs: npt.NDArray[np.complex_ | np.float_],
    *,
    axis: int | tuple[int, ...],
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    Find the coefficients within the shannon number to speed up computations.

    Args:
        wav_coeffs: The wavelet coefficients.
        axis: The axis to search over.

    Returns:
        The non-zero wavelet coefficients.
    """
    return wav_coeffs[wav_coeffs.any(axis=axis)]
