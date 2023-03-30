import numpy as np

from sleplet import logger
from sleplet.functions.fp import SlepianWaveletCoefficientsAfrica
from sleplet.plotting import PlotSphere
from sleplet.region import Region
from sleplet.slepian_methods import slepian_inverse
from sleplet.wavelet_methods import slepian_wavelet_inverse

B = 3
J_MIN = 2
L = 128
NORMALISE = False
SMOOTHING = 2


def main() -> None:
    """
    the reconstruction of a signal in Slepian space
    """
    region = Region(mask_name="africa")
    swc = SlepianWaveletCoefficientsAfrica(
        L, B=B, j_min=J_MIN, region=region, smoothing=SMOOTHING
    )

    # plot
    f_p = np.zeros(swc.slepian.N, dtype=np.complex_)
    for p, coeff in enumerate(swc.wavelet_coefficients):
        logger.info(f"plot reconstruction: {p}")
        f_p += slepian_wavelet_inverse(coeff, swc.wavelets, swc.slepian.N)
        f = slepian_inverse(f_p, L, swc.slepian)
        name = f"africa_wavelet_reconstruction_progressive_{p}_L{L}"
        PlotSphere(f, L, name, normalise=NORMALISE, region=swc.region).execute()


if __name__ == "__main__":
    main()
