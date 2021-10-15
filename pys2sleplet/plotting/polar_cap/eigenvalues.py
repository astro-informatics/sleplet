from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.plot_methods import save_plot

L = 16
THETA_MAX = 40

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")


def main() -> None:
    """
    creates a plot of Slepian eigenvalues against rank
    """
    slepian = SlepianPolarCap(L, np.deg2rad(THETA_MAX))
    orders = np.abs(slepian.order)
    labels = np.array([f"$\pm${m}" if m != 0 else m for m in orders])
    idx = np.argsort(orders)
    scatter = sns.scatterplot(
        x=np.arange(L ** 2) + 1,
        y=slepian.eigenvalues,
        hue=labels,
        hue_order=labels[idx],
        legend="full",
        style=labels,
    )
    scatter.set(xscale="log")
    plt.axvline(x=slepian.N, c="k", ls="--", alpha=0.5)
    plt.annotate(
        f"N={slepian.N}",
        xy=(slepian.N, 1),
        xytext=(0, 15),
        ha="center",
        textcoords="offset points",
        annotation_clip=False,
    )
    ticks = 2 ** np.arange(np.log2(L ** 2) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel("rank $p$")
    plt.ylabel("eigenvalue $\mu$")
    plt.legend(ncol=4)
    save_plot(fig_path, "polar_cap_eigenvalues")


if __name__ == "__main__":
    main()
