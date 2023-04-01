from sleplet.functions import SlepianWaveletCoefficientsAfrica
from sleplet.plotting import PlotSphere
from sleplet.slepian import Region
from sleplet.slepian_methods import slepian_inverse
from sleplet.wavelet_methods import slepian_wavelet_inverse

B = 3
J_MIN = 2
L = 128
NORMALISE = False
SMOOTHING = 2


def main() -> None:
    """The reconstruction of a signal in Slepian space."""
    region = Region(mask_name="africa")
    swc = SlepianWaveletCoefficientsAfrica(
        L,
        B=B,
        j_min=J_MIN,
        region=region,
        smoothing=SMOOTHING,
    )
    f_p = slepian_wavelet_inverse(swc.wavelet_coefficients, swc.wavelets, swc.slepian.N)

    # plot
    f = slepian_inverse(f_p, L, swc.slepian)
    name = f"africa_wavelet_reconstruction_L{L}"
    PlotSphere(f, L, name, normalise=NORMALISE, region=swc.region).execute()


if __name__ == "__main__":
    main()
