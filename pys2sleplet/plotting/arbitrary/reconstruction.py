import pyssht as ssht

from pys2sleplet.flm.maps.south_america import SouthAmerica
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.utils.harmonic_methods import invert_flm_boosted
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import slepian_inverse
from pys2sleplet.utils.vars import SAMPLING_SCHEME

L = 16


def main() -> None:
    """
    the reconstruction of a signal in Slepian space
    """
    region = Region(mask_name="south_america")
    south_america = SouthAmerica(L, region=region)
    sd = SlepianDecomposition(south_america)
    f_p = sd.decompose_all()
    f = slepian_inverse(L, f_p, sd.s_p_lms, coefficients=sd.N)
    flm = ssht.forward(f, L, Method=SAMPLING_SCHEME)
    resolution = calc_plot_resolution(L)
    field = invert_flm_boosted(flm, L, resolution)
    name = f"south_america_reconstruction_L{L}_res{resolution}"
    Plot(field.real, resolution, name).execute()


if __name__ == "__main__":
    main()
