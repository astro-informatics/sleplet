from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle

from pys2sleplet.plotting.polar_cap.inputs import TEXT_BOX, L
from pys2sleplet.plotting.polar_cap.utils import sort_and_clean_df
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger

LEGEND_POS = (0, 0)
ORDERS = 15
RANKS = 60
THETA_RANGE = {10: (0, 0), 20: (0, 1), 30: (1, 0), 40: (1, 1)}

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set()


def main() -> None:
    """
    creates a plot of Slepian coefficient against rank
    """
    N = len(THETA_RANGE) // 2
    _, ax = plt.subplots(N, N, sharex="col", sharey="row")
    for theta_max, position in THETA_RANGE.items():
        _create_plot(ax, position, theta_max)
    ax[LEGEND_POS].legend(ncol=2)

    plt.tight_layout()
    if config.SAVE_FIG:
        for file_type in ["png", "pdf"]:
            filename = fig_path / file_type / f"simons_5-3.{file_type}"
            plt.savefig(filename, bbox_inches="tight")
    if config.AUTO_OPEN:
        plt.show()


def _create_plot(ax: np.ndarray, position: Tuple[int, int], theta_max: int) -> None:
    """
    helper method which actually makes the plot
    """
    df = _create_eigenvalues(theta_max)
    axs = ax[position]
    legend = "full" if position == LEGEND_POS else False
    sns.scatterplot(
        x=df.index,
        y=df["lam"],
        hue=df["m"],
        hue_order=df.sort_values(by="order")["m"],
        legend=legend,
        style=df["m"],
        markers=MarkerStyle.filled_markers[:ORDERS],
        ax=axs,
    )
    if position[1] == 0:
        axs.set_ylabel("$\lambda$")
    if position[0] == 1:
        axs.set_xlabel("rank")
    axs.text(
        0.04,
        0.34,
        fr"$\Theta$={theta_max}$^\circ$",
        transform=axs.transAxes,
        fontsize=12,
        bbox=TEXT_BOX,
    )


def _create_eigenvalues(theta_max: int) -> pd.DataFrame:
    """
    calculates all Slepian coefficients for all orders and sorts them
    """
    df = pd.DataFrame()
    for order in range(-(L - 1), L):
        logger.info(f"calculating theta_max={theta_max}, order={order}")
        eigenvalues = _helper(theta_max, order)
        df_tmp = pd.DataFrame()
        df_tmp["lam"] = eigenvalues
        df_tmp["order"] = abs(order)
        df = pd.concat([df, df_tmp], ignore_index=True)
    df = sort_and_clean_df(df, ORDERS, RANKS, "lam")
    return df


def _helper(theta_max: int, order: int) -> np.ndarray:
    """
    computes the Slepian coefficients for the given order
    """
    slepian = SlepianPolarCap(L, np.deg2rad(theta_max), order=order)
    return slepian.eigenvalues


if __name__ == "__main__":
    main()
