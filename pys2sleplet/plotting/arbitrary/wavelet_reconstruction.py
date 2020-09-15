import pyssht as ssht

from pys2sleplet.flm.kernels.slepian_wavelets import SlepianWavelets
from pys2sleplet.flm.maps.south_america import SouthAmerica
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.wavelet_methods import wavelet_forward, wavelet_inverse

L = 64


def main() -> None:
    """
    the reconstruction of a signal in Slepian space
    """
    south_america = SouthAmerica(L)
    region = Region(mask_name="south_america")
    sw = SlepianWavelets(L, region=region)
    wav_coeffs = wavelet_forward(south_america, sw.wavelets)
    flm = wavelet_inverse(wav_coeffs, sw.wavelets)
    f = ssht.inverse(flm, L)
    resolution = calc_plot_resolution(L)
    name = f"south_america_wavelet_reconstruction_L{L}_res{resolution}"
    Plot(f, L, resolution, name).execute()


if __name__ == "__main__":
    main()
