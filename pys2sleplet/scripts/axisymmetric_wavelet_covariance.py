import numpy as np
import pyssht as ssht
from numpy.random import default_rng

from pys2sleplet.functions.flm.axisymmetric_wavelets import AxisymmetricWavelets
from pys2sleplet.utils.bool_methods import is_ergodic
from pys2sleplet.utils.harmonic_methods import compute_random_signal
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.vars import RANDOM_SEED, SAMPLING_SCHEME
from pys2sleplet.utils.wavelet_methods import (
    axisymmetric_wavelet_forward,
    compute_wavelet_covariance,
)

B = 3
J_MIN = 2
L = 128


def axisymmetric_wavelet_covariance(
    L: int, B: int, j_min: int, *, runs: int = 100, var_flm: float = 1
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
    logger.info(f"L={L}, B={B}, j_min={j_min}")

    # compute wavelets
    aw = AxisymmetricWavelets(L, B=B, j_min=j_min)

    # theoretical covariance
    covar_theory = compute_wavelet_covariance(L, aw.wavelets, var_signal=var_flm)

    # initialise matrix
    covar_runs_shape = (runs,) + covar_theory.shape
    covar_data = np.zeros(covar_runs_shape, dtype=np.complex_)

    # set seed
    rng = default_rng(RANDOM_SEED)

    for i in range(runs):
        logger.info(f"start run: {i+1}/{runs}")

        # Generate normally distributed random complex signal
        flm = compute_random_signal(L, rng, var_signal=var_flm)

        # compute wavelet coefficients
        wlm = axisymmetric_wavelet_forward(L, flm, aw.wavelets)

        # compute covariance from data
        for j, coefficient in enumerate(wlm):
            f_wav_j = ssht.inverse(coefficient, L, Method=SAMPLING_SCHEME)
            covar_data[i, j] = (
                f_wav_j.var() if is_ergodic(j_min, j=j) else f_wav_j[0, 0]
            )

    # compute mean and variance
    runs_axis = 0
    mean_covar_data = covar_data.mean(axis=runs_axis)
    std_covar_data = covar_data.std(axis=runs_axis)

    # override for scaling function
    if not is_ergodic(j_min):
        mean_covar_data[0] = covar_data[0].var()

    # ensure reality
    mean_covar_data = np.abs(mean_covar_data)

    # compute errors
    error_absolute = np.abs(mean_covar_data - covar_theory)
    error_in_std = error_absolute / std_covar_data

    # report errors
    for j in range(len(aw.wavelets)):
        message = (
            f"error in std: {error_in_std[j]:e}"
            if is_ergodic(j_min, j=j)
            else f"absolute error: {error_absolute[j]:e}"
        )
        logger.info(f"axisymmetric wavelet covariance {j}: '{message}'")


if __name__ == "__main__":
    axisymmetric_wavelet_covariance(L, B, J_MIN)
