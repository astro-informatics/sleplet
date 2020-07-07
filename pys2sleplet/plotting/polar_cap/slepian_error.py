from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.plotting.polar_cap.polar_inputs import (
    ALPHA,
    SECOND_COLOUR,
    THETA_MAX,
    L,
)
from pys2sleplet.plotting.polar_cap.utils import (
    create_table,
    earth_region_slepian_coefficients,
)
from pys2sleplet.utils.config import settings

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set()


def main() -> None:
    """
    creates a plot of Slepian coefficient against rank
    """
    df_region = create_table(_helper_region, L, THETA_MAX)
    df_sphere = create_table(_helper_sphere, L, THETA_MAX)
    sns.scatterplot(
        x=df_region.index, y=df_region["qty"], label="region", linewidth=0, marker="."
    )
    sns.scatterplot(
        x=df_sphere.index,
        y=df_sphere["qty"],
        alpha=ALPHA,
        color=SECOND_COLOUR,
        label="sphere",
        linewidth=0,
        marker="*",
    )
    plt.xlabel("coefficient")
    plt.ylabel("relative error")
    plt.yscale("log")
    plt.ylim([1e-17, 1e13])

    plt.tight_layout()
    if settings.SAVE_FIG:
        for file_type in ["png", "pdf"]:
            filename = (
                fig_path
                / file_type
                / f"fp_error_earth_polar{THETA_MAX}_L{L}.{file_type}"
            )
            plt.savefig(filename, bbox_inches="tight")
    if settings.AUTO_OPEN:
        plt.show()


def _helper_sphere(L: int, theta_max: int, order: int) -> np.ndarray:
    """
    the difference in Slepian coefficients by integration of whole sphere
    """
    output = earth_region_slepian_coefficients(
        L, theta_max, order, method="integrate_sphere"
    )
    desired = earth_region_slepian_coefficients(L, theta_max, order)
    error = np.abs(output - desired) / desired
    return error


def _helper_region(L: int, theta_max: int, order: int) -> np.ndarray:
    """
    the difference in Slepian coefficients by integration of region on the sphere
    """
    output = earth_region_slepian_coefficients(
        L, theta_max, order, method="integrate_region"
    )
    desired = earth_region_slepian_coefficients(L, theta_max, order)
    error = np.abs(output - desired) / desired
    return error


if __name__ == "__main__":
    main()
