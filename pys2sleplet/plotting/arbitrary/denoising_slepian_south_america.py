from argparse import ArgumentParser

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
    fun = "south_america"
    region = Region(mask_name=fun)
    f, annotations = denoising_slepian(
        f"slepian_{fun}", L, B, J_MIN, sigma, region, snr
    )
    name = f"{fun}_snr{snr}_n{sigma}_denoised_slepian_L{L}"
    Plot(f, L, name, annotations=annotations).execute()


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
