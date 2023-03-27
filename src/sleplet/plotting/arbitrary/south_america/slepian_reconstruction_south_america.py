from __future__ import annotations

from sleplet.functions.fp.slepian_south_america import SlepianSouthAmerica
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import choose_slepian_method, slepian_inverse
from sleplet.utils.vars import SMOOTHING

L = 128
NORMALISE = False


def main() -> None:
    """
    the reconstruction of a signal in Slepian space
    """
    region = Region(mask_name="south_america")
    slepian = choose_slepian_method(L, region)
    south_america = SlepianSouthAmerica(L, region=region, smoothing=SMOOTHING)

    # perform reconstruction
    f = slepian_inverse(south_america.coefficients, L, slepian)

    # plot
    name = f"south_america_slepian_reconstruction_L{L}"
    Plot(f, L, name, normalise=NORMALISE, region=slepian.region).execute()


if __name__ == "__main__":
    main()
