from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from pys2sleplet.plotting.inputs import PHI_0, PHI_1, THETA_0, THETA_1
from pys2sleplet.slepian.slepian_region.slepian_limit_lat_lon import SlepianLimitLatLon
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import save_plot
from pys2sleplet.utils.slepian_methods import integrate_whole_matrix_slepian_functions

L = 16
RESOLUTION_RANGE = {16: (0, 0), 24: (0, 1), 32: (1, 0), 40: (1, 1)}

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")


def main(zoom: bool = False):
    """
    plots the error matrix of integrating all ranks of Slepian functions
    """
    x = len(RESOLUTION_RANGE) // 2
    _, ax = plt.subplots(x, x, sharex="col", sharey="row")
    for resolution, position in RESOLUTION_RANGE.items():
        _create_plot(ax, position, resolution, zoom)
    save_plot(fig_path, f"integration_error_sphere{'_zoom' if zoom else ''}")


def _create_plot(
    ax: np.ndarray, position: Tuple[int, int], resolution: int, zoom: bool
) -> None:
    """
    helper method which actually makes the plot
    """
    logger.info(f"resolution={resolution}")
    error, N = _helper(L, resolution, THETA_0, THETA_1, PHI_0, PHI_1)
    axs = ax[position]
    axs.set_title(r"$L_{\mathrm{eff}}=%s$" % resolution)
    if position[1] == 0:
        axs.set_ylabel("p")
    if position[0] == 1:
        axs.set_xlabel("p")

    if not zoom:
        axs.axvline(x=N, color="w")
        axs.axhline(y=N, color="w")
        region = L * L
        ticks: Union[str, List] = _create_ticks(L, N)
    else:
        ticks = "auto"
        region = N

    sns.heatmap(
        error[:region, :region],
        ax=axs,
        norm=LogNorm(),
        square=True,
        xticklabels=ticks,
        yticklabels=ticks,
    )


def _helper(
    L: int, resolution: int, theta_min: int, theta_max: int, phi_min: int, phi_max: int
) -> Tuple[np.ndarray, int]:
    """
    calculates the error matrix to plot and Shannon number
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
    return error, slepian.shannon


def _create_ticks(L: int, N: int) -> List[str]:
    """
    create custom tick labels for the plot
    """
    ticks = [""] * (L * L)
    ticks[N - 1] = "N"
    return ticks


if __name__ == "__main__":
    main()
    main(zoom=True)
