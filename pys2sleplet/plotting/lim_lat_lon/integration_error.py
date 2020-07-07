from pathlib import Path
from typing import Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.plotting.inputs import PHI_0, PHI_1, THETA_0, THETA_1
from pys2sleplet.slepian.slepian_region.slepian_limit_lat_lon import SlepianLimitLatLon
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import integrate_whole_matrix_slepian_functions

L = 16
RESOLUTION_RANGE = {16: (0, 0), 24: (0, 1), 32: (1, 0), 40: (1, 1)}

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set()


def main():
    """
    plots the error matrix of integrating all ranks of Slepian functions
    """
    N = len(RESOLUTION_RANGE) // 2
    _, ax = plt.subplots(N, N, sharex="col", sharey="row")
    for resolution, position in RESOLUTION_RANGE.items():
        _create_plot(ax, position, resolution)

    plt.tight_layout()
    if settings.SAVE_FIG:
        for file_type in ["png", "pdf"]:
            filename = fig_path / file_type / f"integration_error.{file_type}"
            plt.savefig(filename, bbox_inches="tight")
    if settings.AUTO_OPEN:
        plt.show()


def _create_plot(ax: np.ndarray, position: Tuple[int, int], resolution: int) -> None:
    """
    helper method which actually makes the plot
    """
    logger.info(f"resolution={resolution}")
    error = _helper(L, resolution, THETA_0, THETA_1, PHI_0, PHI_1)
    axs = ax[position]
    sns.heatmap(error, ax=axs, square=True, xticklabels=False, yticklabels=False)
    axs.set_title(r"$L_{\mathrm{effective}}=%s$" % resolution)


def _helper(
    L: int, resolution: int, theta_min: int, theta_max: int, phi_min: int, phi_max: int
) -> np.ndarray:
    """
    calculates the error matrix to plot
    """
    slepian = SlepianLimitLatLon(
        L,
        theta_min=np.deg2rad(theta_min),
        theta_max=np.deg2rad(theta_max),
        phi_min=np.deg2rad(phi_min),
        phi_max=np.deg2rad(phi_max),
    )
    output = integrate_whole_matrix_slepian_functions(
        slepian.eigenvectors, L, resolution
    )
    desired = np.identity(output.shape[0])
    error = np.abs(output - desired)
    return error


if __name__ == "__main__":
    main()
