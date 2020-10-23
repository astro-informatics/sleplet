from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.plotting.inputs import THETA_MAX
from pys2sleplet.plotting.plotting_utils import (
    earth_region_slepian_coefficients,
    get_shannon,
)
from pys2sleplet.utils.plot_methods import save_plot

L = 19

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")


def main() -> None:
    """
    creates a plot of Slepian coefficients against rank
    """
    region = _helper_region(L, THETA_MAX)
    sphere = _helper_sphere(L, THETA_MAX)
    N = get_shannon(L, THETA_MAX)
    ax = plt.gca()
    sns.scatterplot(
        x=range(N), y=region, ax=ax, label="region", linewidth=0, marker="."
    )
    sns.scatterplot(
        x=range(N), y=sphere, ax=ax, label="sphere", linewidth=0, marker="*"
    )
    ax.set_xlabel("coefficients")
    ax.set_ylabel("relative error")
    ax.set_yscale("log")
    save_plot(fig_path, f"fp_error_earth_polar{THETA_MAX}_L{L}")


def _helper_sphere(L: int, theta_max: int) -> np.ndarray:
    """
    the difference in Slepian coefficients by integration of whole sphere
    """
    output = earth_region_slepian_coefficients(L, theta_max, method="integrate_sphere")
    desired = earth_region_slepian_coefficients(L, theta_max)
    return np.abs(output - desired) / desired


def _helper_region(L: int, theta_max: int) -> np.ndarray:
    """
    the difference in Slepian coefficients by integration of region on the sphere
    """
    output = earth_region_slepian_coefficients(L, theta_max, method="integrate_region")
    desired = earth_region_slepian_coefficients(L, theta_max)
    return np.abs(output - desired) / desired


if __name__ == "__main__":
    main()
