from pys2sleplet.functions.flm.axisymmetric_wavelets import AxisymmetricWavelets
from pys2sleplet.functions.flm.earth import Earth
from pys2sleplet.plotting.create_plot_sphere import Plot
from pys2sleplet.utils.denoising import denoising_axisym
from pys2sleplet.utils.plot_methods import find_max_amplitude

B = 2
J_MIN = 0
L = 128
N_SIGMA = 3
PLOT_TYPE = "real"
SNR_IN = 10


def main() -> None:
    """
    reproduce the denoising demo from s2let paper
    """
    # create map & noised map
    fun = Earth(L)
    fun_noised = Earth(L, noise=SNR_IN)

    # create wavelets
    aw = AxisymmetricWavelets(L, B=B, j_min=J_MIN)

    # fix amplitude
    amplitude = find_max_amplitude(fun_noised, PLOT_TYPE)

    f, _, _ = denoising_axisym(fun, fun_noised, aw, SNR_IN, N_SIGMA)
    name = f"{fun.name}_denoised_axisym"
    Plot(f, L, name, amplitude=amplitude, plot_type=PLOT_TYPE).execute()


if __name__ == "__main__":
    main()
