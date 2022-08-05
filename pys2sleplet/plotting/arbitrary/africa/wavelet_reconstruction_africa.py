from pys2sleplet.functions.fp.slepian_wavelet_coefficients_africa import (
    SlepianWaveletCoefficientsAfrica,
)
from pys2sleplet.plotting.create_plot_sphere import Plot
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
    region = Region(mask_name="africa")
    swc = SlepianWaveletCoefficientsAfrica(
        L, B=B, j_min=J_MIN, region=region, smoothing=SMOOTHING
    )
    f_p = slepian_wavelet_inverse(swc.wavelet_coefficients, swc.wavelets, swc.slepian.N)

    # plot
    f = slepian_inverse(f_p, L, swc.slepian)
    name = f"africa_wavelet_reconstruction_L{L}"
    Plot(f, L, name, normalise=NORMALISE, region=swc.region).execute()


if __name__ == "__main__":
    main()
