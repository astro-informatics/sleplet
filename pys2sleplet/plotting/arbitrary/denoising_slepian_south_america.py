from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.denoising import denoising_slepian
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.region import Region

B = 2
J_MIN = 0
L = 64
N_SIGMA = 3


def main() -> None:
    """
    denoising demo using Slepian wavelets
    """
    fun = "south_america"
    region = Region(mask_name=fun)
    f = denoising_slepian(fun, L, B, J_MIN, N_SIGMA, region)
    resolution = calc_plot_resolution(L)
    name = f"{fun}_denoised_slepian_L{L}_res{resolution}"
    Plot(f, L, resolution, name).execute()


if __name__ == "__main__":
    main()
