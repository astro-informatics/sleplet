from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.utils.plot_methods import save_plot

file_location = Path(__file__).resolve()
data_path = file_location.parents[2] / "data" / "slepian" / "eigensolutions"
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")

FULL_SPECTRUM = True
L = 128
SHANNON = 690


def main() -> None:
    """
    plots the tiling of the Slepian line
    """
    # read in eigenvalues
    file_path = f"D_south_america_L{L}"
    if not FULL_SPECTRUM:
        file_path += f"_N{SHANNON}"
    eigenvalues = np.load(data_path / file_path / "eigenvalues.npy")

    # produce plot
    p_limit = len(eigenvalues)
    p_range = np.arange(0, p_limit)
    plt.semilogx(p_range, eigenvalues, ".")

    # add Slepian line annotation
    if FULL_SPECTRUM:
        plt.axvline(SHANNON, color="k", linestyle="dashed")
        plt.annotate(
            f"N={SHANNON}",
            xy=(SHANNON, 1),
            xytext=(17, 3),
            ha="center",
            textcoords="offset points",
            annotation_clip=False,
        )

    # adjust ticks
    ticks = 2 ** np.arange(np.log2(p_limit) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$p$")
    plt.ylabel(r"$\mu$")
    output = f"south_america_eigenvalues_L{L}"
    if FULL_SPECTRUM:
        output += "_full"
    save_plot(fig_path, output)


if __name__ == "__main__":
    main()
