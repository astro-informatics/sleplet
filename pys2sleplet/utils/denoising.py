from typing import Dict, List, Tuple

import numpy as np
import pyssht as ssht

from pys2sleplet.functions.flm.axisymmetric_wavelets import AxisymmetricWavelets
from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from pys2sleplet.utils.function_dicts import MAPS
from pys2sleplet.utils.noise import (
    compute_sigma_j,
    compute_snr,
    harmonic_hard_thresholding,
    slepian_hard_thresholding,
)
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import slepian_forward, slepian_inverse
from pys2sleplet.utils.wavelet_methods import (
    axisymmetric_wavelet_forward,
    axisymmetric_wavelet_inverse,
    slepian_wavelet_forward,
    slepian_wavelet_inverse,
)


def denoising_axisym(
    name: str, L: int, B: int, j_min: int, n_sigma: int, snr_in: int
) -> Tuple[np.ndarray, float, float]:
    """
    reproduce the denoising demo from s2let paper
    """
    # create map & noised map
    fun = MAPS[name](L)
    fun_noised = MAPS[name](L, noise=snr_in)

    # create wavelets
    aw = AxisymmetricWavelets(L, B=B, j_min=j_min)

    # compute wavelet coefficients
    w = axisymmetric_wavelet_forward(L, fun_noised.coefficients, aw.wavelets)

    # compute wavelet noise
    sigma_j = compute_sigma_j(L, fun.coefficients, aw.wavelets[1:], snr_in)

    # hard thresholding
    w_denoised = harmonic_hard_thresholding(L, w, sigma_j, n_sigma)

    # wavelet synthesis
    flm = axisymmetric_wavelet_inverse(L, w_denoised, aw.wavelets)
    f = ssht.inverse(flm, L)

    # compute SNR
    noised_snr = compute_snr(
        L, fun.coefficients, fun_noised.coefficients - fun.coefficients
    )
    denoised_snr = compute_snr(L, fun.coefficients, flm - fun.coefficients)
    return f, noised_snr, denoised_snr


def denoising_slepian(
    name: str, L: int, B: int, j_min: int, n_sigma: int, region: Region, snr_in: int
) -> Tuple[np.ndarray, List[Dict]]:
    """
    denoising demo using Slepian wavelets
    """
    # create map & noised map
    fun = MAPS[name](L)
    fun_noised = MAPS[name](L, noise=snr_in)

    # create wavelets
    sw = SlepianWavelets(L, B=B, j_min=j_min, region=region)

    # compute Slepian coefficients
    fun_p = slepian_forward(L, fun.coefficients, sw.slepian)
    fun_noised_p = slepian_forward(L, fun_noised.coefficients, sw.slepian)

    # compute wavelet noise
    sigma_j = compute_sigma_j(L, fun_p, sw.wavelets[1:], snr_in)

    # compute wavelet coefficients
    w = slepian_wavelet_forward(fun_noised_p, sw.wavelets, sw.slepian.N)

    # hard thresholding
    w_denoised = slepian_hard_thresholding(L, w, sigma_j, n_sigma, sw.slepian)

    # wavelet synthesis
    f_p = slepian_wavelet_inverse(w_denoised, sw.wavelets, sw.slepian.N)
    f = slepian_inverse(L, f_p, sw.slepian)

    # compute SNR
    compute_snr(L, fun_p, fun_noised_p - fun_p)
    compute_snr(L, fun_p, f_p - fun_p)
    return f, sw.annotations
