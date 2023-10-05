"""Methods to work with wavelet and wavelet coefficients."""
import numpy as np
import numpy.typing as npt

import pyssht as ssht

import sleplet._convolution_methods
import sleplet.slepian_methods


def slepian_wavelet_forward(
    f_p: npt.NDArray[np.complex_ | np.float_],
    wavelets: npt.NDArray[np.float_],
    shannon: int,
) -> npt.NDArray[np.complex_ | np.float_]:
    """
    Computes the coefficients of the given tiling function in Slepian space.

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
    Computes the inverse wavelet transform in Slepian space.

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
        wav_coeffs.T,
        shannon=shannon,
    )
    return wavelet_reconstruction.sum(axis=0)


def axisymmetric_wavelet_forward(
    L: int,
    flm: npt.NDArray[np.complex_ | np.float_],
    wavelets: npt.NDArray[np.complex_],
) -> npt.NDArray[np.complex_]:
    """
    Computes the coefficients of the axisymmetric wavelets.

    Args:
        L: The spherical harmonic bandlimit.
        flm: The spherical harmonic coefficients.
        wavelets: Axisymmetric wavelets.

    Returns:
        Axisymmetric wavelets coefficients.
    """
    wavelets = np.array([s2fft.samples.flm_1d_to_2d(wav, L) for wav in wavelets])
    w = np.zeros(wavelets.shape, dtype=np.complex_)
    for ell in range(L):
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * wavelets[:, ell, L - 1].conj()
        for m in range(-ell, ell + 1):
            w[:, ell, L - 1 + m] = wav_0 * flm[ssht.elm2ind(ell, m)]
    return np.array([s2fft.samples.flm_2d_to_1d(wav, L) for wav in w])


def axisymmetric_wavelet_inverse(
    L: int,
    wav_coeffs: npt.NDArray[np.complex_],
    wavelets: npt.NDArray[np.complex_],
) -> npt.NDArray[np.complex_]:
    """
    Computes the inverse axisymmetric wavelet transform.

    Args:
        L: The spherical harmonic bandlimit.
        wav_coeffs: Axisymmetric wavelet coefficients.
        wavelets: Axisymmetric wavelets.

    Returns:
        Spherical harmonic coefficients of the signal.
    """
    flm = np.zeros(s2fft.samples.flm_shape(L), dtype=np.complex_)
    wavelets = np.array([s2fft.samples.flm_1d_to_2d(w, L) for w in wavelets])
    wav_coeffs = np.array([s2fft.samples.flm_1d_to_2d(wc, L) for wc in wav_coeffs])
    for ell in range(L):
        wav_0 = np.sqrt((4 * np.pi) / (2 * ell + 1)) * wavelets[:, ell, L - 1]
        for m in range(-ell, ell + 1):
            flm[ell, L - 1 + m] = (wav_coeffs[:, ell, L - 1 + m] * wav_0).sum()
    return flm


def _create_axisymmetric_wavelets(
    L: int,
    B: int,
    j_min: int,
) -> npt.NDArray[np.complex_]:
    """Computes the axisymmetric wavelets."""
    kappas = create_kappas(L, B, j_min)
    wavelets = np.zeros(
        (kappas.shape[0], *s2fft.samples.flm_shape(L)),
        dtype=np.complex_,
    )
    for ell in range(L):
        factor = np.sqrt((2 * ell + 1) / (4 * np.pi))
        wavelets[:, ell, L - 1] = factor * kappas[:, ell]
    return np.array([s2fft.samples.flm_2d_to_1d(wav, L) for wav in wavelets])


def create_kappas(xlim: int, B: int, j_min: int) -> npt.NDArray[np.float_]:
    r"""
    Computes the Slepian wavelets.

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
    Finds the coefficients within the shannon number to speed up computations.

    Args:
        wav_coeffs: The wavelet coefficients.
        axis: The axis to search over.

    Returns:
        The non-zero wavelet coefficients.
    """
    return wav_coeffs[wav_coeffs.any(axis=axis)]
