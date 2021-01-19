from pys2sleplet.functions.flm.axisymmetric_wavelets import AxisymmetricWavelets
from pys2sleplet.functions.flm.south_america import SouthAmerica
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
    # create map & noised map
    fun = SouthAmerica(L)
    fun_noised = SouthAmerica(L, noise=SNR_IN)

    # create wavelets
    aw = AxisymmetricWavelets(L, B=B, j_min=J_MIN)

    f, _, _ = denoising_axisym(fun, fun_noised, aw, SNR_IN, N_SIGMA)
    name = f"{fun.name}_denoised_axisym_L{L}"
    Plot(f, L, name).execute()


if __name__ == "__main__":
    main()
