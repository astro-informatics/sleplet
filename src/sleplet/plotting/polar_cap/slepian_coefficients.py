from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sleplet.plotting.inputs import THETA_MAX
from sleplet.plotting.plotting_utils import (
    earth_region_harmonic_coefficients,
    earth_region_slepian_coefficients,
)
from sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from sleplet.utils.plot_methods import save_plot

L = 16

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")


def main() -> None:
    """
    creates a plot of Slepian coefficients against rank
    """
    N = SlepianPolarCap(L, np.deg2rad(THETA_MAX)).N
    flm = earth_region_harmonic_coefficients(L, THETA_MAX)[:N]
    f_p = np.sort(earth_region_slepian_coefficients(L, THETA_MAX))[::-1]
    ax = plt.gca()
    sns.scatterplot(x=range(N), y=f_p, ax=ax, label="slepian", linewidth=0, marker="*")
    sns.scatterplot(x=range(N), y=flm, ax=ax, label="harmonic", linewidth=0, marker=".")
    ax.set_xlabel("coefficients")
    ax.set_ylabel("magnitude")
    save_plot(fig_path, f"fp_earth_polar{THETA_MAX}_L{L}")


if __name__ == "__main__":
    main()
