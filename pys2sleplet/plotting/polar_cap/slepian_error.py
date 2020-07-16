from pathlib import Path
from typing import Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.plotting.inputs import TEXT_BOX, THETA_MAX
from pys2sleplet.plotting.polar_cap.utils import (
    create_table,
    earth_region_slepian_coefficients,
    get_shannon,
)
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.plot_methods import save_plot

L = 19

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")


def main() -> None:
    """
    creates a plot of Slepian coefficient against rank
    """
    df_region = create_table(_helper_region, L, THETA_MAX)
    df_sphere = create_table(_helper_sphere, L, THETA_MAX)
    N = get_shannon(L, THETA_MAX)
    ax = plt.gca()
    sns.scatterplot(
        x=df_region.index,
        y=df_region["other"],
        ax=ax,
        label="region",
        linewidth=0,
        marker=".",
    )
    sns.scatterplot(
        x=df_sphere.index,
        y=df_sphere["other"],
        ax=ax,
        label="sphere",
        linewidth=0,
        marker="*",
    )
    ax.axvline(x=N - 1, color="k")
    ax.text(0.17, 0.93, f"N={N}", transform=ax.transAxes, bbox=TEXT_BOX)
    ax.legend(loc=(0.8, 0.6))
    ax.set_xlabel("coefficient")
    ax.set_ylabel("relative error")
    ax.set_yscale("log")
    ax.set_ylim([1e-17, 1e13])
    save_plot(fig_path, f"fp_error_earth_polar{THETA_MAX}_L{L}")


def _helper_sphere(L: int, theta_max: int, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    the difference in Slepian coefficients by integration of whole sphere
    """
    output = earth_region_slepian_coefficients(
        L, theta_max, order, method="integrate_sphere"
    )
    desired = earth_region_slepian_coefficients(L, theta_max, order)
    error = np.abs(output - desired) / desired
    ev = SlepianPolarCap(L, np.deg2rad(theta_max), order=order).eigenvalues
    return ev, error


def _helper_region(L: int, theta_max: int, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    the difference in Slepian coefficients by integration of region on the sphere
    """
    output = earth_region_slepian_coefficients(
        L, theta_max, order, method="integrate_region"
    )
    desired = earth_region_slepian_coefficients(L, theta_max, order)
    error = np.abs(output - desired) / desired
    ev = SlepianPolarCap(L, np.deg2rad(theta_max), order=order).eigenvalues
    return ev, error


if __name__ == "__main__":
    main()
