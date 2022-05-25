from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.functions.flm.axisymmetric_wavelet_coefficients_south_america import (
    AxisymmetricWaveletCoefficientsSouthAmerica,
)
from pys2sleplet.functions.fp.slepian_wavelet_coefficients_south_america import (
    SlepianWaveletCoefficientsSouthAmerica,
)
from pys2sleplet.utils.plot_methods import save_plot
from pys2sleplet.utils.region import Region

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")

B = 3
J_MIN = 2
L = 128
STEP = 0.01


def _plot_slepian_coefficients() -> int:
    """
    plot the Slepian wavelet coefficients for the South America region
    """
    # initialise wavelet coefficients
    region = Region(mask_name="south_america")
    swc = SlepianWaveletCoefficientsSouthAmerica(L, B=B, j_min=J_MIN, region=region)

    # find sorted coefficients
    w_p = np.sort(np.abs(swc.wavelet_coefficients.sum(axis=0)))[::-1]

    # perform plot
    plt.plot(w_p, label=r"$|W^{{\varphi}}_{p}|$")

    return swc.slepian.N


def _plot_axisymmetric_coefficients(shannon: int) -> None:
    """
    plot the axisymmetric wavelet coefficients for the South America region
    """
    # initialise wavelet coefficients
    awc = AxisymmetricWaveletCoefficientsSouthAmerica(L, B=B, j_min=J_MIN)

    # find sorted coefficients
    w_lm = np.sort(np.abs(awc.wavelet_coefficients.sum(axis=0)))[::-1]

    # perform plot
    plt.plot(w_lm[:shannon], label=r"$|W^{{\varphi}}_{\ell m}|$")


def main() -> None:
    """
    Plot a comparison of the absolute values of the wavelet coefficients
    compared to the Slepian coefficients. Expect the Slepian coefficients to
    decay faster than the wavelet coefficients.
    """
    shannon = _plot_slepian_coefficients()
    _plot_axisymmetric_coefficients(shannon)
    plt.legend()
    plt.xlabel("coefficient index")
    save_plot(fig_path, f"south_america_sparsity_wavelet_coefficients_comparison_L{L}")


if __name__ == "__main__":
    main()
