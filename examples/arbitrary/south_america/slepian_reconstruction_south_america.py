from sleplet.functions import SlepianSouthAmerica
from sleplet.plotting import PlotSphere
from sleplet.slepian import Region
from sleplet.slepian_methods import choose_slepian_method, slepian_inverse

L = 128
NORMALISE = False
SMOOTHING = 2


def main() -> None:
    """The reconstruction of a signal in Slepian space."""
    region = Region(mask_name="south_america")
    slepian = choose_slepian_method(L, region)
    south_america = SlepianSouthAmerica(L, region=region, smoothing=SMOOTHING)

    # perform reconstruction
    f = slepian_inverse(south_america.coefficients, L, slepian)

    # plot
    name = f"south_america_slepian_reconstruction_L{L}"
    PlotSphere(f, L, name, normalise=NORMALISE, region=slepian.region).execute()


if __name__ == "__main__":
    main()
