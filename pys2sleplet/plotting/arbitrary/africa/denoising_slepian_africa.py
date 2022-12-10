from argparse import ArgumentParser

from sleplet.functions.fp.slepian_africa import SlepianAfrica
from sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.denoising import denoising_slepian_wavelet
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
    region = Region(mask_name="africa")

    # create map & noised map
    fun = SlepianAfrica(L, region=region, smoothing=SMOOTHING)
    fun_noised = SlepianAfrica(L, noise=snr, region=region, smoothing=SMOOTHING)

    # create wavelets
    sw = SlepianWavelets(L, B=B, j_min=J_MIN, region=region)

    # fix amplitude
    amplitude = find_max_amplitude(fun)

    f = denoising_slepian_wavelet(fun, fun_noised, sw, snr, sigma)
    name = f"{fun.name}{filename_args(snr, 'snr')}{filename_args(sigma,'n')}_denoised"
    Plot(
        f, L, name, amplitude=amplitude, normalise=NORMALISE, region=sw.region
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
        type=int,
        default=N_SIGMA,
    )
    args = parser.parse_args()
    main(args.noise, args.sigma)
