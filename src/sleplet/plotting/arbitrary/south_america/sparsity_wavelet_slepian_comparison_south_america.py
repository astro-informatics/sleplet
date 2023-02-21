from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sleplet.functions.flm.axisymmetric_wavelet_coefficients_south_america import (
    AxisymmetricWaveletCoefficientsSouthAmerica,
)
from sleplet.functions.fp.slepian_wavelet_coefficients_south_america import (
    SlepianWaveletCoefficientsSouthAmerica,
)
from sleplet.utils.plot_methods import save_plot
from sleplet.utils.region import Region

file_location = Path(__file__).resolve()
fig_path = file_location.parents[3] / "figures"
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
    w_p = np.sort(np.abs(swc.wavelet_coefficients), axis=1)[:, ::-1]

    # perform plot
    plt.plot(w_p[0], label=r"$|W^{{\phi}}_{p}|$")

    for j, w_j in enumerate(w_p[1:]):
        plt.plot(w_j, label=rf"$|W^{{\phi^{j+J_MIN}}}_{{p}}|$")

    return swc.slepian.N


def _plot_axisymmetric_coefficients(shannon: int) -> None:
    """
    plot the axisymmetric wavelet coefficients for the South America region
    """
    # initialise wavelet coefficients
    awc = AxisymmetricWaveletCoefficientsSouthAmerica(L, B=B, j_min=J_MIN)

    # find sorted coefficients
    w_lm = np.sort(np.abs(awc.wavelet_coefficients), axis=1)[:, ::-1]

    # perform plot
    plt.plot(w_lm[0, :shannon], "--", label=r"$|W^{{\phi}}_{\ell m}|$")

    for j, w_j in enumerate(w_lm[1:]):
        plt.plot(w_j[:shannon], "--", label=rf"$|W^{{\phi^{j+J_MIN}}}_{{\ell m}}|$")


def main(limit: bool) -> None:
    """
    Plot a comparison of the absolute values of the wavelet coefficients
    compared to the Slepian coefficients. Expect the Slepian coefficients to
    decay faster than the wavelet coefficients.
    """
    shannon = _plot_slepian_coefficients()
    _plot_axisymmetric_coefficients(shannon)
    plt.xlabel("coefficient index")
    plt.legend()
    filename = f"south_america_sparsity_wavelet_coefficients_comparison_L{L}"
    if limit:
        plt.xlim(right=300)
        plt.ylim(top=50)
        filename += "_limit"
    save_plot(fig_path, filename)


if __name__ == "__main__":
    parser = ArgumentParser(description="South America sparsity")
    parser.add_argument(
        "--limit",
        "-l",
        action="store_true",
        help="flag which limits the region of the plot",
    )
    args = parser.parse_args()
    main(args.limit)