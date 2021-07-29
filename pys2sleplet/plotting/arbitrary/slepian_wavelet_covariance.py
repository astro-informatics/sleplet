import numpy as np
from numpy.random import default_rng

from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from pys2sleplet.plotting.create_plot_sphere import Plot
from pys2sleplet.utils.harmonic_methods import compute_random_signal
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import slepian_inverse
from pys2sleplet.utils.vars import RANDOM_SEED
from pys2sleplet.utils.wavelet_methods import (
    compute_slepian_wavelet_covariance,
    find_non_zero_wavelet_coefficients,
    slepian_wavelet_forward,
)

B = 3
J_MIN = 2
L = 128
NORMALISE = False
RUNS = 10
VAR_FP = 1


def main() -> None:
    """
    plots the difference between the theoretical &
    experimental covariances for the Slepian wavelets
    """
    # compute wavelets
    region = Region(mask_name="south_america")
    sw = SlepianWavelets(L, B=B, j_min=J_MIN, region=region)

    # theoretical covariance
    covar_theory = compute_slepian_wavelet_covariance(
        sw.wavelets, VAR_FP, L, sw.slepian
    )

    # initialise matrix
    covar_runs_shape = (RUNS,) + covar_theory.shape
    covar_data_runs = np.zeros(covar_runs_shape, dtype=np.complex_)

    # set seed
    rng = default_rng(RANDOM_SEED)

    for i in range(RUNS):
        # Generate normally distributed random complex signal
        f_p = compute_random_signal(L, rng, VAR_FP)

        # compute wavelet coefficients
        w_p = slepian_wavelet_forward(f_p, sw.wavelets, sw.slepian.N)

        # compute field values
        for j, coefficient in enumerate(w_p):
            logger.info(f"run: {i+1}/{RUNS}, compute covariance: {j+1}/{len(w_p)}")
            covar_data_runs[i, j] = slepian_inverse(coefficient, L, sw.slepian)

    # compute covariance
    runs_axis = 0
    covar_data = covar_data_runs.var(axis=runs_axis)

    # compute differences
    omega_axis = (1, 2)
    non_zero_theory = find_non_zero_wavelet_coefficients(covar_theory, omega_axis)
    non_zero_data = find_non_zero_wavelet_coefficients(covar_data, omega_axis)
    differences = np.abs(non_zero_data - non_zero_theory) / non_zero_theory

    # report errors
    for j, diff in enumerate(differences):
        name = f"slepian_covariance_diff_{j}"
        logger.info(name)
        Plot(diff, L, name, normalise=NORMALISE, region=sw.region).execute()


if __name__ == "__main__":
    main()
