from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.denoising import denoising_axisym

B = 3
J_MIN = 2
L = 128
N_SIGMA = 2
SNR_IN = -10


def main() -> None:
    """
    denoising demo of South America to compare to Slepian wavelet case
    """
    fun = "south_america"
    f, _, _ = denoising_axisym(fun, L, B, J_MIN, N_SIGMA, SNR_IN)
    name = f"{fun}_denoised_axisym_L{L}"
    Plot(f, L, name).execute()


if __name__ == "__main__":
    main()
