from pys2sleplet.functions.flm.south_america import SouthAmerica
from pys2sleplet.functions.fp.slepian_wavelets import SlepianWavelets
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import slepian_forward, slepian_inverse
from pys2sleplet.utils.wavelet_methods import slepian_wavelet_forward

L = 128


def main() -> None:
    """
    the wavelet coefficients of a Slepian region
    """
    region = Region(mask_name="south_america")
    south_america = SouthAmerica(L, region=region)
    sw = SlepianWavelets(L, region=region)
    coefficients = slepian_forward(L, south_america.coefficients, sw.slepian)
    wav_coeffs = slepian_wavelet_forward(coefficients, sw.wavelets)

    # plot
    for p, coeff in enumerate(wav_coeffs):
        logger.info(f"plot coefficients: {p}")
        f = slepian_inverse(L, coeff, sw.slepian)
        resolution = calc_plot_resolution(L)
        name = f"south_america_wavelet_coefficient_{p}_L{L}_res{resolution}"
        Plot(f, L, resolution, name).execute()


if __name__ == "__main__":
    main()
