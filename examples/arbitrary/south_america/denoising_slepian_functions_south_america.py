import argparse

import numpy as np
import numpy.typing as npt

import sleplet

B = 3
J_MIN = 2
L = 128
N_SIGMA = 2
NORMALISE = False
SMOOTHING = 2
SNR_IN = -10


def _denoising_slepian_function(
    signal: sleplet.functions.SlepianSouthAmerica,
    noised_signal: sleplet.functions.SlepianSouthAmerica,
    snr_in: float,
    n_sigma: int,
) -> npt.NDArray[np.complex128]:
    """Denoising demo using Slepian function."""
    # compute Slepian noise
    sigma_noise = sleplet.noise.compute_sigma_noise(
        signal.coefficients,
        snr_in,
        denominator=signal.L**2,
    )

    # hard thresholding
    f_p = sleplet.noise.slepian_function_hard_thresholding(
        signal.L,
        noised_signal.coefficients,
        sigma_noise,
        n_sigma,
        signal.slepian,
    )

    # compute SNR
    sleplet.noise.compute_snr(signal.coefficients, f_p - signal.coefficients, "Slepian")

    return sleplet.slepian_methods.slepian_inverse(f_p, signal.L, signal.slepian)


def main(snr: float, sigma: int) -> None:
    """Denoising demo using Slepian wavelets."""
    print(f"SNR={snr}, n_sigma={sigma}")
    # setup
    region = sleplet.slepian.Region(mask_name="south_america")

    # create map & noised map
    fun = sleplet.functions.SlepianSouthAmerica(L, region=region, smoothing=SMOOTHING)
    fun_noised = sleplet.functions.SlepianSouthAmerica(
        L,
        noise=snr,
        region=region,
        smoothing=SMOOTHING,
    )

    # fix amplitude
    amplitude = sleplet.plot_methods.find_max_amplitude(fun)

    f = _denoising_slepian_function(fun, fun_noised, snr, sigma)
    name = f"{fun.name}_{snr}snr_{sigma}n_denoised_function"
    sleplet.plotting.PlotSphere(
        f,
        L,
        name,
        amplitude=amplitude,
        normalise=NORMALISE,
        region=region,
    ).execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="denoising")
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
