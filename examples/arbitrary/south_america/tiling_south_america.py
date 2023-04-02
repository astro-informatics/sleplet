import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import pchip

from sleplet.plot_methods import save_plot
from sleplet.slepian import SlepianArbitrary
from sleplet.wavelet_methods import create_kappas

sns.set(context="paper")

B = 3
J_MIN = 2
L = 128
STEP = 0.01


def main() -> None:
    """Plots the tiling of the Slepian line."""
    xlim = L**2
    x = np.arange(xlim)
    xi = np.arange(0, xlim - 1 + STEP, STEP)
    kappas = create_kappas(xlim, B, J_MIN)
    yi = pchip(x, kappas[0])
    plt.semilogx(xi, yi(xi), label=r"$\Phi_p$")
    for j, k in enumerate(kappas[1:]):
        yi = pchip(x, k)
        plt.semilogx(xi, yi(xi), label=rf"$\Psi^{{{j+J_MIN}}}_p$")
    slepian = SlepianArbitrary(L, "south_america")
    plt.axvline(slepian.N, color="k", linestyle="dashed")
    plt.annotate(
        f"N={slepian.N}",
        xy=(slepian.N, 1),
        xytext=(17, 3),
        ha="center",
        textcoords="offset points",
        annotation_clip=False,
    )
    plt.xlim(1, xlim)
    ticks = 2 ** np.arange(np.log2(xlim) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$p$")
    plt.legend()
    save_plot(f"south_america_slepian_tiling_L{L}")


if __name__ == "__main__":
    main()
