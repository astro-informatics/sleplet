import numpy as np
from numpy.random import default_rng

from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from pys2sleplet.utils.harmonic_methods import compute_random_signal
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import slepian_inverse
from pys2sleplet.utils.vars import RANDOM_SEED
from pys2sleplet.utils.wavelet_methods import (
    compute_slepian_wavelet_covariance,
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
    covar_w_theory = compute_slepian_wavelet_covariance(
        sw.wavelets, var_fp, L, sw.slepian
    )

    # initialise matrix
    covar_runs_shape = (runs,) + covar_w_theory.shape
    covar_w_data_runs = np.zeros(covar_runs_shape, dtype=np.complex_)

    # set seed
    rng = default_rng(RANDOM_SEED)

    for i in range(runs):
        # Generate normally distributed random complex signal
        f_p = compute_random_signal(L, rng, var_fp)

        # compute wavelet coefficients
        w_p = slepian_wavelet_forward(f_p, sw.wavelets, sw.slepian.N)

        # compute field values
        for j, coefficient in enumerate(w_p):
            logger.info(f"run: {i+1}/{runs}, compute covariance: {j+1}/{len(w_p)}")
            covar_w_data_runs[i, j] = slepian_inverse(L, coefficient, sw.slepian)

    # define axes
    runs_axis, theta_axis, phi_axis = 0, 1, 2

    # compute covariance
    covar_w_data = covar_w_data_runs.var(axis=runs_axis)

    # compute errors
    w_error_absolute = np.abs(covar_w_data - covar_w_theory).mean(
        axis=(theta_axis, phi_axis)
    )

    # report errors
    for j in range(len(w_p)):
        logger.info(
            f"slepian wavelet covariance {j}: absolute error: {w_error_absolute[j]:e}"
        )


if __name__ == "__main__":
    region = Region(mask_name="south_america")
    slepian_wavelet_covariance(L=128, B=3, j_min=2, region=region)
