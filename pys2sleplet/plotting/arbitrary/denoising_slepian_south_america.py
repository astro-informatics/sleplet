from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.denoising import denoising_slepian
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.region import Region

B = 3
J_MIN = 2
L = 128
N_SIGMA = 3
SNR_IN = 10


def main() -> None:
    """
    denoising demo using Slepian wavelets
    """
    fun = "south_america"
    region = Region(mask_name=fun)
    f, annotations = denoising_slepian(
        f"slepian_{fun}", L, B, J_MIN, N_SIGMA, region, SNR_IN
    )
    resolution = calc_plot_resolution(L)
    name = f"{fun}_denoised_slepian_L{L}_res{resolution}"
    Plot(f, L, resolution, name, annotations=annotations).execute()


if __name__ == "__main__":
    main()
