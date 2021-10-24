from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.slepian.slepian_region.slepian_arbitrary import SlepianArbitrary
from pys2sleplet.utils.plot_methods import save_plot

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")

L = 128


def main() -> None:
    """
    plots the tiling of the Slepian line
    """
    slepian = SlepianArbitrary(L, "south_america")
    p_range = np.arange(0, slepian.N)
    plt.semilogx(p_range, slepian.eigenvalues[: slepian.N], "k.")
    ticks = 2 ** np.arange(np.log2(slepian.N) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$p$")
    plt.ylabel(r"$\mu$")
    save_plot(fig_path, f"south_america_eigenvalues_L{L}")


if __name__ == "__main__":
    main()
