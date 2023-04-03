from argparse import ArgumentParser

import numpy as np
from numpy import typing as npt

from sleplet.functions import SlepianSouthAmerica
from sleplet.noise import (
    compute_sigma_noise,
    compute_snr,
    slepian_function_hard_thresholding,
)
from sleplet.plot_methods import find_max_amplitude
from sleplet.plotting import PlotSphere
from sleplet.slepian import Region
from sleplet.slepian_methods import slepian_inverse

B = 3
J_MIN = 2
L = 128
N_SIGMA = 2
NORMALISE = False
SMOOTHING = 2
SNR_IN = -10


def _denoising_slepian_function(
    signal: SlepianSouthAmerica,
    noised_signal: SlepianSouthAmerica,
    snr_in: float,
    n_sigma: int,
) -> npt.NDArray[np.complex_]:
    """Denoising demo using Slepian function."""
    # compute Slepian noise
    sigma_noise = compute_sigma_noise(
        signal.coefficients,
        snr_in,
        denominator=signal.L**2,
    )

    # hard thresholding
    f_p = slepian_function_hard_thresholding(
        signal.L,
        noised_signal.coefficients,
        sigma_noise,
        n_sigma,
        signal.slepian,
    )

    # compute SNR
    compute_snr(signal.coefficients, f_p - signal.coefficients, "Slepian")

    return slepian_inverse(f_p, signal.L, signal.slepian)


def main(snr: float, sigma: int) -> None:
    """Denoising demo using Slepian wavelets."""
    print(f"SNR={snr}, n_sigma={sigma}")
    # setup
    region = Region(mask_name="south_america")

    # create map & noised map
    fun = SlepianSouthAmerica(L, region=region, smoothing=SMOOTHING)
    fun_noised = SlepianSouthAmerica(L, noise=snr, region=region, smoothing=SMOOTHING)

    # fix amplitude
    amplitude = find_max_amplitude(fun)

    f = _denoising_slepian_function(fun, fun_noised, snr, sigma)
    name = f"{fun.name}_{snr}snr_{sigma}n_denoised_function"
    PlotSphere(
        f,
        L,
        name,
        amplitude=amplitude,
        normalise=NORMALISE,
        region=region,
    ).execute()


if __name__ == "__main__":
    parser = ArgumentParser(description="denoising")
    parser.add_argument(
        "--noise",
        "-n",
        type=int,
        default=SNR_IN,
    )
    parser.add_argument(
        "--sigma",
        "-s",
        type=float,
        default=N_SIGMA,
    )
    args = parser.parse_args()
    main(args.noise, args.sigma)
