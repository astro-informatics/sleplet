import pyssht as ssht

from pys2sleplet.flm.kernels.slepian_wavelets import SlepianWavelets
from pys2sleplet.flm.maps.south_america import SouthAmerica
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.vars import SAMPLING_SCHEME
from pys2sleplet.utils.wavelet_methods import wavelet_inverse

B = 2
J_MIN = 0
L = 64


def main() -> None:
    """
    the reconstruction of a signal in Slepian space
    """
    south_america = SouthAmerica(L)
    region = Region(mask_name="south_america")
    scaling = SlepianWavelets(L, B=B, j_min=J_MIN, region=region)
    flm = wavelet_inverse(south_america, scaling.multipole)
    for j in range(scaling.j_max - J_MIN):
        wavelet = SlepianWavelets(L, B=B, j_min=J_MIN, j=j, region=region)
        flm += wavelet_inverse(south_america, wavelet.multipole)
    f = ssht.inverse(flm, L, Method=SAMPLING_SCHEME)
    resolution = calc_plot_resolution(L)
    name = f"south_america_wavelet_reconstruction_L{L}_res{resolution}"
    Plot(f, L, resolution, name).execute()


if __name__ == "__main__":
    main()
