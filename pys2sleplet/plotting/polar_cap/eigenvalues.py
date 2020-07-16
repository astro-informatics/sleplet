from pathlib import Path
from typing import Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle

from pys2sleplet.plotting.inputs import FIGSIZE, LINEWIDTH, TEXT_BOX
from pys2sleplet.plotting.polar_cap.utils import create_table, get_shannon
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.logger import logger

L = 19
LEGEND_POS = (0, 0)
ORDERS = 15
RANKS = 60
THETA_RANGE = {10: (0, 0), 20: (0, 1), 30: (1, 0), 40: (1, 1)}

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set()


def main() -> None:
    """
    creates a plot of Slepian eigenvalues against rank
    """
    N = len(THETA_RANGE) // 2
    _, ax = plt.subplots(N, N, sharex="col", sharey="row", figsize=FIGSIZE)
    for theta_max, position in THETA_RANGE.items():
        _create_plot(ax, position, theta_max)
    ax[LEGEND_POS].legend(ncol=2)

    plt.tight_layout()
    if settings.SAVE_FIG:
        for file_type in ["png", "pdf"]:
            filename = fig_path / file_type / f"simons_5_3.{file_type}"
            plt.savefig(filename, bbox_inches="tight")
    if settings.AUTO_OPEN:
        plt.show()


def _create_plot(ax: np.ndarray, position: Tuple[int, int], theta_max: int) -> None:
    """
    helper method which actually makes the plot
    """
    logger.info(f"theta_max={theta_max}")
    df = create_table(_helper, L, theta_max, ORDERS, RANKS)
    N = get_shannon(L, theta_max)
    axs = ax[position]
    legend = "full" if position == LEGEND_POS else False
    sns.scatterplot(
        x=df.index,
        y=df["qty"],
        hue=df["m"],
        hue_order=df.sort_values(by="order")["m"],
        legend=legend,
        style=df["m"],
        markers=MarkerStyle.filled_markers[:ORDERS],
        ax=axs,
    )
    axs.axvline(x=N - 1, linewidth=LINEWIDTH)
    axs.annotate(
        f"N={N}",
        xy=(N - 1, 1),
        xytext=(0, 7),
        ha="center",
        textcoords="offset points",
        annotation_clip=False,
    )
    if position[1] == 0:
        axs.set_ylabel("eigenvalue $\lambda$")
    if position[0] == 1:
        axs.set_xlabel("rank")
    axs.text(
        0.03,
        0.34,
        fr"$\Theta$={theta_max}$^\circ$",
        transform=axs.transAxes,
        fontsize=12,
        bbox=TEXT_BOX,
    )


def _helper(L: int, theta_max: int, order: int) -> np.ndarray:
    """
    computes the Slepian eigenvalues for the given order
    """
    slepian = SlepianPolarCap(L, np.deg2rad(theta_max), order=order)
    return slepian.eigenvalues


if __name__ == "__main__":
    main()
