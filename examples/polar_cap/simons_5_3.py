import typing

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

import sleplet

sns.set(context="paper")

L = 16
LEGEND_POS = (0, 0)
RANKS = 60
TEXT_BOX: dict[str, str | float] = {"boxstyle": "round", "color": "w"}
THETA_RANGE = {10: (0, 0), 20: (0, 1), 30: (1, 0), 40: (1, 1)}


def main() -> None:
    """Create a plot of Slepian eigenvalues against rank."""
    x = len(THETA_RANGE) // 2
    _, ax = plt.subplots(x, x, sharex="col", sharey="row")
    for theta_max, position in THETA_RANGE.items():
        _create_plot(ax, position, theta_max)
    ax[LEGEND_POS].legend(ncol=2)
    print("Opening: simons_5_3")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


def _create_plot(
    ax: npt.NDArray[typing.Any],
    position: tuple[int, int],
    theta_max: int,
) -> None:
    """Create the plot."""
    print(f"theta_max={theta_max}")
    slepian = sleplet.slepian.SlepianPolarCap(L, np.deg2rad(theta_max))
    axs = ax[position]
    legend = "full" if position == LEGEND_POS else False
    orders = np.abs(slepian.order[:RANKS])
    labels = np.array([f"$\\pm${m}" if m != 0 else m for m in orders])
    idx = np.argsort(orders)
    sns.scatterplot(
        x=range(RANKS),
        y=slepian.eigenvalues[:RANKS],
        hue=labels,
        hue_order=labels[idx],
        legend=legend,
        style=labels,
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
        axs.set_ylabel("$\\mu$")
    if position[0] == 1:
        axs.set_xlabel("$p$")
    axs.text(
        0.02,
        0.34,
        rf"$\Theta={{{theta_max}}}^\circ$",
        transform=axs.transAxes,
        bbox=TEXT_BOX,
    )


if __name__ == "__main__":
    main()
