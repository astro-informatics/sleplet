from pys2sleplet.flm.maps.south_america import SouthAmerica
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import (
    choose_slepian_method,
    slepian_forward,
    slepian_inverse,
)

L = 64


def main() -> None:
    """
    the reconstruction of a signal in Slepian space
    """
    region = Region(mask_name="south_america")
    slepian = choose_slepian_method(L, region)
    south_america = SouthAmerica(L, region=region)

    # perform reconstruction
    f_p = slepian_forward(L, south_america.multipole, slepian)
    f = slepian_inverse(L, f_p, slepian)

    # plot
    resolution = calc_plot_resolution(L)
    name = f"south_america_slepian_reconstruction_L{L}_res{resolution}"
    Plot(f, L, resolution, name).execute()


if __name__ == "__main__":
    main()
