import pathlib
import sys

import numpy as np

import sleplet

sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from _denoising_axisym import denoising_axisym  # noqa: E402

B = 2
J_MIN = 0
L = 128
N_SIGMA = 3
NORMALISE = False
SNR_IN = 10


def main() -> None:
    """Reproduce the denoising demo from S2LET paper."""
    # create map & noised map
    fun = sleplet.functions.Earth(L)
    fun_noised = sleplet.functions.Earth(L, noise=SNR_IN)

    # create wavelets
    aw = sleplet.functions.AxisymmetricWavelets(L, B=B, j_min=J_MIN)

    # fix amplitude
    amplitude = sleplet.plot_methods.find_max_amplitude(fun)

    f, noised_snr, denoised_snr = denoising_axisym(
        fun,
        fun_noised,
        aw,
        SNR_IN,
        N_SIGMA,
        rotate_to_south_america=True,
    )
    np.testing.assert_array_less(noised_snr, denoised_snr)
    name = f"{fun.name}_denoised_axisym"
    sleplet.plotting.PlotSphere(
        f,
        L,
        name,
        amplitude=amplitude,
        normalise=NORMALISE,
    ).execute()


if __name__ == "__main__":
    main()
