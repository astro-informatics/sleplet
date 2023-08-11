import argparse
import pathlib
import sys

import sleplet

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from _denoising_slepian_wavelet import denoising_slepian_wavelet  # noqa: E402

B = 3
J_MIN = 2
L = 128
N_SIGMA = 2
NORMALISE = False
SMOOTHING = 2
SNR_IN = -10


def main(snr: float, sigma: int) -> None:
    """Denoising demo using Slepian wavelets."""
    print(f"SNR={snr}, n_sigma={sigma}")
    # setup
    region = sleplet.slepian.Region(mask_name="africa")

    # create map & noised map
    fun = sleplet.functions.SlepianAfrica(L, region=region, smoothing=SMOOTHING)
    fun_noised = sleplet.functions.SlepianAfrica(
        L,
        noise=snr,
        region=region,
        smoothing=SMOOTHING,
    )

    # create wavelets
    sw = sleplet.functions.SlepianWavelets(L, B=B, j_min=J_MIN, region=region)

    # fix amplitude
    amplitude = sleplet.plot_methods.find_max_amplitude(fun)

    f = denoising_slepian_wavelet(fun, fun_noised, sw, snr, sigma)
    name = f"{fun.name}_{snr}snr_{sigma}n_denoised"
    sleplet.plotting.PlotSphere(
        f,
        L,
        name,
        amplitude=amplitude,
        normalise=NORMALISE,
        region=sw.region,
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
        type=int,
        default=N_SIGMA,
    )
    args = parser.parse_args()
    main(args.noise, args.sigma)
