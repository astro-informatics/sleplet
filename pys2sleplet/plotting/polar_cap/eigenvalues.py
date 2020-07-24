from pathlib import Path
from typing import Tuple

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle

from pys2sleplet.plotting.inputs import TEXT_BOX
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import save_plot

L = 19
LEGEND_POS = (0, 0)
RANKS = 60
THETA_RANGE = {10: (0, 0), 20: (0, 1), 30: (1, 0), 40: (1, 1)}

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")


def main() -> None:
    """
    creates a plot of Slepian eigenvalues against rank
    """
    x = len(THETA_RANGE) // 2
    _, ax = plt.subplots(x, x, sharex="col", sharey="row")
    for theta_max, position in THETA_RANGE.items():
        _create_plot(ax, position, theta_max)
    ax[LEGEND_POS].legend(ncol=2)
    save_plot(fig_path, "simons_5_3")


def _create_plot(ax: np.ndarray, position: Tuple[int, int], theta_max: int) -> None:
    """
    helper method which actually makes the plot
    """
    logger.info(f"theta_max={theta_max}")
    slepian = SlepianPolarCap(L, np.deg2rad(theta_max))
    axs = ax[position]
    legend = "full" if position == LEGEND_POS else False
    orders = np.abs(slepian.order[:RANKS])
    labels = np.array([f"$\pm${m}" if m != 0 else m for m in orders])
    idx = np.argsort(orders)
    sns.scatterplot(
        x=range(RANKS),
        y=slepian.eigenvalues[:RANKS],
        hue=labels,
        hue_order=labels[idx],
        legend=legend,
        style=labels,
        markers=MarkerStyle.filled_markers,
        ax=axs,
    )
    axs.axvline(x=slepian.N - 1)
    axs.annotate(
        f"N={slepian.N}",
        xy=(slepian.N - 1, 1),
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
        0.02,
        0.34,
        fr"$\Theta$={theta_max}$^\circ$",
        transform=axs.transAxes,
        bbox=TEXT_BOX,
    )


if __name__ == "__main__":
    main()
