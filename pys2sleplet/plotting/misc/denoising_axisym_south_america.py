from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.denoising import denoising_axisym
from pys2sleplet.utils.plot_methods import calc_plot_resolution

B = 3
J_MIN = 2
L = 128
N_SIGMA = 3
SNR_IN = 1


def main() -> None:
    """
    denoising demo of South America to compare to Slepian wavelet case
    """
    fun = "south_america"
    f, _, _ = denoising_axisym(fun, L, B, J_MIN, N_SIGMA, SNR_IN)
    resolution = calc_plot_resolution(L)
    name = f"{fun}_denoised_axisym_L{L}_res{resolution}"
    Plot(f, L, resolution, name).execute()


if __name__ == "__main__":
    main()
