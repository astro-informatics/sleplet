from argparse import ArgumentParser

from pys2sleplet.functions.fp.slepian_south_america import SlepianSouthAmerica
from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.denoising import denoising_slepian
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import find_max_amplitude
from pys2sleplet.utils.region import Region

B = 3
J_MIN = 2
L = 128
N_SIGMA = 2
PLOT_TYPE = "real"
SMOOTHED = True
SNR_IN = -10


def main(snr: int, sigma: int) -> None:
    """
    denoising demo using Slepian wavelets
    """
    logger.info(f"SNR={snr}, n_sigma={sigma}")
    # setup
    region = Region(mask_name="south_america")

    # create map & noised map
    fun = SlepianSouthAmerica(L, region=region, smoothed=SMOOTHED)
    fun_noised = SlepianSouthAmerica(L, noise=snr, region=region, smoothed=SMOOTHED)

    # create wavelets
    sw = SlepianWavelets(L, B=B, j_min=J_MIN, region=region)

    # fix amplitude
    amplitude = find_max_amplitude(
        L, fun_noised.coefficients, PLOT_TYPE, slepian=sw.slepian
    )

    f = denoising_slepian(fun, fun_noised, sw, snr, sigma)
    name = f"{fun.name}_snr{snr}_n{sigma}_denoised_L{L}"
    Plot(
        f, L, name, amplitude=amplitude, plot_type=PLOT_TYPE, region=sw.region
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
