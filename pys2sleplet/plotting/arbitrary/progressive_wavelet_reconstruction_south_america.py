import numpy as np

from pys2sleplet.functions.fp.slepian_wavelet_coefficients_south_america import (
    SlepianWaveletCoefficientsSouthAmerica,
)
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import slepian_inverse
from pys2sleplet.utils.wavelet_methods import slepian_wavelet_inverse

L = 128


def main() -> None:
    """
    the reconstruction of a signal in Slepian space
    """
    region = Region(mask_name="south_america")
    swc = SlepianWaveletCoefficientsSouthAmerica(L, region=region)

    # plot
    f_p = np.zeros(swc.slepian.N, dtype=np.complex128)
    for p, coeff in enumerate(swc.wavelet_coefficients):
        logger.info(f"plot reconstruction: {p}")
        f_p += slepian_wavelet_inverse(coeff, swc.wavelets, swc.slepian.N)
        f = slepian_inverse(L, f_p, swc.slepian)
        resolution = calc_plot_resolution(L)
        name = (
            f"south_america_wavelet_reconstruction_progressive_{p}_L{L}_res{resolution}"
        )
        Plot(f, L, resolution, name, annotations=swc.annotations).execute()


if __name__ == "__main__":
    main()
