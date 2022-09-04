import numpy as np

from pys2sleplet.functions.fp.slepian_wavelet_coefficients_south_america import (
    SlepianWaveletCoefficientsSouthAmerica,
)
from pys2sleplet.plotting.create_plot_sphere import Plot
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import slepian_inverse
from pys2sleplet.utils.vars import SMOOTHING
from pys2sleplet.utils.wavelet_methods import slepian_wavelet_inverse

B = 3
J_MIN = 2
L = 128
NORMALISE = False


def main() -> None:
    """
    the reconstruction of a signal in Slepian space
    """
    region = Region(mask_name="south_america")
    swc = SlepianWaveletCoefficientsSouthAmerica(
        L, B=B, j_min=J_MIN, region=region, smoothing=SMOOTHING
    )

    # plot
    f_p = np.zeros(swc.slepian.N, dtype=np.complex_)
    for p, coeff in enumerate(swc.wavelet_coefficients):
        logger.info(f"plot reconstruction: {p}")
        f_p += slepian_wavelet_inverse(coeff, swc.wavelets, swc.slepian.N)
        f = slepian_inverse(f_p, L, swc.slepian)
        name = f"south_america_wavelet_reconstruction_progressive_{p}_L{L}"
        Plot(f, L, name, normalise=NORMALISE, region=swc.region).execute()


if __name__ == "__main__":
    main()