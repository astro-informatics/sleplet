import sys
from pathlib import Path

from sleplet.functions import AxisymmetricWavelets, SouthAmerica
from sleplet.plot_methods import find_max_amplitude
from sleplet.plotting import PlotSphere

sys.path.append(str(Path(__file__).resolve().parent))

from _denoising_axisym import denoising_axisym  # noqa: E402

B = 2
J_MIN = 0
L = 128
N_SIGMA = 3
NORMALISE = False
SNR_IN = 10


def main() -> None:
    """Contrast denosiing with an Earth map versus South America map."""
    # create map & noised map
    fun = SouthAmerica(L)
    fun_noised = SouthAmerica(L, noise=SNR_IN)

    # create wavelets
    aw = AxisymmetricWavelets(L, B=B, j_min=J_MIN)

    # fix amplitude
    amplitude = find_max_amplitude(fun)

    f, _, _ = denoising_axisym(fun, fun_noised, aw, SNR_IN, N_SIGMA)
    name = f"{fun.name}_denoised_axisym"
    PlotSphere(f, L, name, amplitude=amplitude, normalise=NORMALISE).execute()


if __name__ == "__main__":
    main()
