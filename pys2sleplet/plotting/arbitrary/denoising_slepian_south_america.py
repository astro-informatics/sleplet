from argparse import ArgumentParser

from pys2sleplet.functions.fp.slepian_south_america import SlepianSouthAmerica
from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.denoising import denoising_slepian
from pys2sleplet.utils.region import Region

B = 3
J_MIN = 2
L = 128
N_SIGMA = 2
SNR_IN = -10


def main(snr: int, sigma: float) -> None:
    """
    denoising demo using Slepian wavelets
    """
    # setup
    region = Region(mask_name="south_america")

    # create map & noised map
    fun = SlepianSouthAmerica(L, region=region)
    fun_noised = SlepianSouthAmerica(L, region=region, noise=snr)

    # create wavelets
    sw = SlepianWavelets(L, B=B, j_min=J_MIN, region=region)

    f = denoising_slepian(fun, fun_noised, sw, snr, sigma)
    name = f"{fun.name}_snr{snr}_n{sigma}_denoised_L{L}"
    Plot(f, L, name, annotations=sw.annotations).execute()


if __name__ == "__main__":
    parser = ArgumentParser(description="denoising")
    parser.add_argument(
        "--snr",
        "-s",
        type=int,
        default=SNR_IN,
    )
    parser.add_argument(
        "--sigma",
        "-n",
        type=float,
        default=N_SIGMA,
    )
    args = parser.parse_args()
    main(args.snr, args.sigma)
