import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sleplet

sns.set(context="paper")

COLOURS = ["b", "k"]
L = 128
REGIONS = ["africa", "south_america"]
X_POS = [19, -16]
ZORDER = [2, 1]


def main() -> None:
    """Plots the tiling of the Slepian line."""
    for c, region in enumerate(REGIONS):
        slepian = sleplet.slepian.SlepianArbitrary(L, region)
        p_range = np.arange(0, L**2)
        plt.semilogx(
            p_range,
            slepian.eigenvalues,
            f"{COLOURS[c]}.",
            label=" ".join(region.title().split("_")),
            zorder=ZORDER[c],
        )
        plt.axvline(slepian.N, color=COLOURS[c], ls="--", alpha=0.8)
        plt.annotate(
            f"N={slepian.N}",
            xy=(slepian.N, 1),
            xytext=(X_POS[c], 3),
            ha="center",
            textcoords="offset points",
            annotation_clip=False,
            color=COLOURS[c],
        )
    ticks = 2 ** np.arange(np.log2(L**2) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$p$")
    plt.ylabel(r"$\mu$")
    plt.legend(loc=3)
    print(f"Opening: combined_eigenvalues_L{L}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


if __name__ == "__main__":
    main()
