import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sleplet

sns.set(context="paper")

L = 128


def main() -> None:
    """Plot the tiling of the Slepian line."""
    slepian = sleplet.slepian.SlepianArbitrary(L, "south_america")
    p_range = np.arange(0, L**2)
    plt.semilogx(p_range, slepian.eigenvalues, "k.")
    plt.axvline(slepian.N, c="k", ls="--", alpha=0.8)
    plt.annotate(
        f"N={slepian.N}",
        xy=(slepian.N, 1),
        xytext=(17, 3),
        ha="center",
        textcoords="offset points",
        annotation_clip=False,
    )
    ticks = 2 ** np.arange(np.log2(L**2) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$p$")
    plt.ylabel(r"$\mu$")
    print(f"Opening: south_america_eigenvalues_L{L}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


if __name__ == "__main__":
    main()
