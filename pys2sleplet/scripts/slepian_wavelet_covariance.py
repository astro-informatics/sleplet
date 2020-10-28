import numpy as np
from numpy.random import default_rng

from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from pys2sleplet.utils.bool_methods import is_ergodic
from pys2sleplet.utils.harmonic_methods import compute_random_signal
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import slepian_inverse
from pys2sleplet.utils.vars import RANDOM_SEED
from pys2sleplet.utils.wavelet_methods import (
    compute_wavelet_covariance,
    slepian_wavelet_forward,
)


def slepian_wavelet_covariance(
    L: int, B: int, j_min: int, region: Region, runs: int = 10, var_fp: float = 1
) -> None:
    """
    compute theoretical covariance of wavelet coefficients

    the covariance <Wj(omega)Wj*(omega)> is given by the following expression:
    sigma^2 Sum(l,0) |Psi^j_l0|^2

    where sigma^2 is the variance of the harmonic coefficients and Psi^j_l0
    are the harmonic coefficients of the j-th wavelet

    a similar expression applies for the scaling function coefficients

    should we use the actual variance of each realisation instead?
    """
    logger.info(f"L={L}, B={B}, j_min={j_min}, region='{region.name_ending}'")

    # compute wavelets
    sw = SlepianWavelets(L, B=B, j_min=j_min, region=region)

    # theoretical covariance
    covar_w_theory = compute_wavelet_covariance(sw.wavelets, var_fp)

    # initialise matrix
    covar_w_data = np.zeros((sw.wavelets.shape[0], runs), dtype=np.complex128)

    # set seed
    rng = default_rng(RANDOM_SEED)

    for i in range(runs):
        logger.info(f"start run: {i}")

        # Generate normally distributed random complex signal
        f_p = compute_random_signal(L, rng, var_fp)

        # compute wavelet coefficients
        w_p = slepian_wavelet_forward(f_p, sw.wavelets, sw.slepian.N)

        # compute covariance from data
        for j in range(sw.wavelets.shape[0]):
            f_wav_j = slepian_inverse(L, w_p[j], sw.slepian)
            covar_w_data[j, i] = (
                f_wav_j.var() if is_ergodic(j_min, j) else f_wav_j[0, 0]
            )

    # compute mean and variance
    mean_covar_w_data = covar_w_data.mean(axis=1)
    std_covar_w_data = covar_w_data.std(axis=1)

    # override for scaling function
    if not is_ergodic(j_min):
        mean_covar_w_data[0] = covar_w_data[0].var()

    # compute errors
    w_error_absolute = np.abs(mean_covar_w_data - covar_w_theory)
    std_covar_w_data = np.where(std_covar_w_data != 0, std_covar_w_data, np.nan)
    w_error_in_std = w_error_absolute / std_covar_w_data

    # report errors
    for j in range(sw.wavelets.shape[0]):
        message = (
            f"error in std: {w_error_in_std[j]:e}"
            if is_ergodic(j_min, j)
            else f"absolute error: {w_error_absolute[j]:e}"
        )
        logger.info(f"slepian wavelet covariance {j}: '{message}'")


if __name__ == "__main__":
    region = Region(mask_name="south_america")
    slepian_wavelet_covariance(L=128, B=3, j_min=2, region=region)
