import sys
from pathlib import Path

import numpy as np
from numpy.random import default_rng

from sleplet.functions import SlepianWavelets
from sleplet.harmonic_methods import compute_random_signal
from sleplet.plotting import PlotSphere
from sleplet.slepian import Region
from sleplet.slepian_methods import slepian_inverse
from sleplet.wavelet_methods import (
    find_non_zero_wavelet_coefficients,
    slepian_wavelet_forward,
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from _slepian_wavelet_covariance import compute_slepian_wavelet_covariance  # noqa: E402

B = 3
J_MIN = 2
L = 128
NORMALISE = False
RANDOM_SEED = 30
RUNS = 10
VAR_FP = 1


def main() -> None:
    """
    Plots the difference between the theoretical &
    experimental covariances for the Slepian wavelets.
    """
    # compute wavelets
    region = Region(mask_name="south_america")
    sw = SlepianWavelets(L, B=B, j_min=J_MIN, region=region)

    # theoretical covariance
    covar_theory = compute_slepian_wavelet_covariance(L, sw, var_signal=VAR_FP)

    # initialise matrix
    covar_runs_shape = (RUNS, *covar_theory.shape)
    covar_data_runs = np.zeros(covar_runs_shape, dtype=np.complex_)

    # set seed
    rng = default_rng(RANDOM_SEED)

    for i in range(RUNS):
        # Generate normally distributed random complex signal
        f_p = compute_random_signal(L, rng, var_signal=VAR_FP)

        # compute wavelet coefficients
        w_p = slepian_wavelet_forward(f_p, sw.wavelets, sw.slepian.N)

        # compute field values
        for j, coefficient in enumerate(w_p):
            print(f"run: {i+1}/{RUNS}, compute covariance: {j+1}/{len(w_p)}")
            covar_data_runs[i, j] = slepian_inverse(coefficient, L, sw.slepian)

    # compute covariance
    covar_data = covar_data_runs.var(axis=0)

    # compute differences
    non_zero_theory = find_non_zero_wavelet_coefficients(covar_theory, axis=(1, 2))
    non_zero_data = find_non_zero_wavelet_coefficients(covar_data, axis=(1, 2))
    differences = np.abs(non_zero_data - non_zero_theory) / non_zero_theory

    # report errors
    for j, diff in enumerate(differences):
        name = f"slepian_covariance_diff_{j}"
        print(name)
        PlotSphere(diff, L, name, normalise=NORMALISE, region=sw.region).execute()


if __name__ == "__main__":
    main()
