from typing import Dict, List, Tuple

import numpy as np
import pyssht as ssht

from pys2sleplet.functions.flm.axisymmetric_wavelets import AxisymmetricWavelets
from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from pys2sleplet.utils.function_dicts import MAPS_LM, MAPS_P
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.noise import (
    compute_sigma_j,
    compute_snr,
    harmonic_hard_thresholding,
    slepian_hard_thresholding,
)
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import slepian_inverse
from pys2sleplet.utils.vars import EARTH_ALPHA, EARTH_BETA, EARTH_GAMMA, SAMPLING_SCHEME
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
    logger.info(f"L={L}, B={B}, J_min={j_min}, n_sigma={n_sigma}, SNR_in={snr_in}")
    # create map & noised map
    fun = MAPS_LM[name](L)
    fun_noised = MAPS_LM[name](L, noise=snr_in)

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

    # rotate to South America
    flm_rot = (
        ssht.rotate_flms(flm, EARTH_ALPHA, EARTH_BETA, EARTH_GAMMA, L)
        if "earth" in name
        else flm
    )

    # real space
    f = ssht.inverse(flm_rot, L, Method=SAMPLING_SCHEME)

    # compute SNR
    denoised_snr = compute_snr(L, fun.coefficients, flm - fun.coefficients)
    return f, fun_noised.snr, denoised_snr


def denoising_slepian(
    name: str, L: int, B: int, j_min: int, n_sigma: int, region: Region, snr_in: int
) -> Tuple[np.ndarray, List[Dict]]:
    """
    denoising demo using Slepian wavelets
    """
    # create map & noised map
    fun = MAPS_P[name](L)
    fun_noised = MAPS_P[name](L, noise=snr_in)

    # create wavelets
    sw = SlepianWavelets(L, B=B, j_min=j_min, region=region)

    # compute wavelet noise
    sigma_j = compute_sigma_j(L, fun.coefficients, sw.wavelets[1:], snr_in)

    # compute wavelet coefficients
    w = slepian_wavelet_forward(fun_noised.coefficients, sw.wavelets, sw.slepian.N)

    # hard thresholding
    w_denoised = slepian_hard_thresholding(L, w, sigma_j, n_sigma, sw.slepian)

    # wavelet synthesis
    f_p = slepian_wavelet_inverse(w_denoised, sw.wavelets, sw.slepian.N)
    f = slepian_inverse(L, f_p, sw.slepian)

    # compute SNR
    compute_snr(L, fun.coefficients, f_p - fun.coefficients)
    return f, sw.annotations
