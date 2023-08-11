import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import sleplet

sns.set(context="paper")

L = 16
THETA_MAX = 40


def main() -> None:
    """Creates a plot of Slepian eigenvalues against rank."""
    slepian = sleplet.slepian.SlepianPolarCap(L, np.deg2rad(THETA_MAX))
    p_range = np.arange(0, L**2)
    plt.semilogx(p_range, slepian.eigenvalues, "k.")
    plt.axvline(x=slepian.N, c="k", ls="--", alpha=0.5)
    plt.annotate(
        f"N={slepian.N}",
        xy=(slepian.N, 1),
        xytext=(0, 15),
        ha="center",
        textcoords="offset points",
        annotation_clip=False,
    )
    ticks = 2 ** np.arange(np.log2(L**2) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$p$")
    plt.ylabel(r"$\mu$")
    print("Opening: polar_cap_eigenvalues")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


if __name__ == "__main__":
    main()
