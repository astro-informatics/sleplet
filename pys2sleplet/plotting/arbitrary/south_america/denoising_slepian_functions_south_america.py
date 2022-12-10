from argparse import ArgumentParser

from sleplet.functions.fp.slepian_south_america import SlepianSouthAmerica
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.denoising import denoising_slepian_function
from sleplet.utils.logger import logger
from sleplet.utils.plot_methods import find_max_amplitude
from sleplet.utils.region import Region
from sleplet.utils.string_methods import filename_args
from sleplet.utils.vars import SMOOTHING

B = 3
J_MIN = 2
L = 128
N_SIGMA = 2
NORMALISE = False
SNR_IN = -10


def main(snr: int, sigma: int) -> None:
    """
    denoising demo using Slepian wavelets
    """
    logger.info(f"SNR={snr}, n_sigma={sigma}")
    # setup
    region = Region(mask_name="south_america")

    # create map & noised map
    fun = SlepianSouthAmerica(L, region=region, smoothing=SMOOTHING)
    fun_noised = SlepianSouthAmerica(L, noise=snr, region=region, smoothing=SMOOTHING)

    # fix amplitude
    amplitude = find_max_amplitude(fun)

    f = denoising_slepian_function(fun, fun_noised, snr, sigma)
    name = (
        f"{fun.name}{filename_args(snr, 'snr')}"
        f"{filename_args(sigma,'n')}_denoised_function"
    )
    Plot(f, L, name, amplitude=amplitude, normalise=NORMALISE, region=region).execute()


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
