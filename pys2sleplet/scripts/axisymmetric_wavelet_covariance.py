import numpy as np
import pyssht as ssht
from numpy.random import default_rng

from pys2sleplet.flm.kernels.axisymmetric_wavelets import AxisymmetricWavelets
from pys2sleplet.utils.bool_methods import is_ergodic
from pys2sleplet.utils.harmonic_methods import compute_random_flm
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.vars import RANDOM_SEED
from pys2sleplet.utils.wavelet_methods import (
    axisymmetric_wavelet_forward,
    compute_wavelet_covariance,
)


def axisymmetric_wavelet_covariance(
    L: int, B: int, j_min: int, runs: int = 100, var_flm: float = 1
) -> None:
    """
    compute theoretical covariance of wavelet coefficients

    the covariance <Wj(omega)Wj*(omega)> is given by the following expression:
    sigma^2 Sum(l,0) |Psi^j_l0|^2

    where sigma^2 is the variance of the harmonic coefficients and Psi^j_l0
    are the harmonic coefficients of the j-th wavelet

    a similar expression applies for the scaling function coefficients

    # should we use the actual variance var(flmn) of each realisation instead?
    """
    logger.info(f"L={L}, B={B}, j_min={j_min}")

    # compute wavelets
    aw = AxisymmetricWavelets(L, B=B, j_min=j_min)

    # theoretical covariance
    covar_w_theory = compute_wavelet_covariance(aw.wavelets, var_flm)

    # initialise matrix
    covar_w_data = np.zeros((aw.wavelets.shape[0], runs), dtype=np.complex128)

    # set seed
    rng = default_rng(RANDOM_SEED)

    for i in range(runs):
        # Generate normally distributed random flmn of complex signal
        flm = compute_random_flm(L, rng, var_flm=var_flm)

        # compute wavelet coefficients
        w = axisymmetric_wavelet_forward(L, flm, aw.wavelets)

        # compute covariance from data
        for j in range(aw.wavelets.shape[0]):
            f_wav_j = ssht.inverse(w[j], L)
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
    w_error_in_std = w_error_absolute / std_covar_w_data

    # report errors
    for j in range(aw.wavelets.shape[0]):
        message = (
            f"error in std: {w_error_in_std[j]:e}"
            if is_ergodic(j_min, j)
            else f"absolute error: {w_error_absolute[j]:e}"
        )
        logger.info(f"axisymmetric wavelet covariance {j}: '{message}'")


if __name__ == "__main__":
    axisymmetric_wavelet_covariance(L=128, B=2, j_min=0)
