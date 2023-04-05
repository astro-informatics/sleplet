import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import pchip

from sleplet.wavelet_methods import create_kappas

sns.set(context="paper")

B = 3
J_MIN = 2
L = 128
STEP = 0.01


def main() -> None:
    """Plots the tiling of the harmonic line."""
    xlim = L
    x = np.arange(xlim)
    xi = np.arange(0, xlim - 1 + STEP, STEP)
    kappas = create_kappas(xlim, B, J_MIN)
    yi = pchip(x, kappas[0])
    plt.semilogx(xi, yi(xi), label=r"$\Phi_{\ell0}$")
    for j, k in enumerate(kappas[1:]):
        yi = pchip(x, k)
        plt.semilogx(xi, yi(xi), label=rf"$\Psi^{{{j+J_MIN}}}_{{\ell0}}$")
    plt.xlim(1, xlim)
    ticks = 2 ** np.arange(np.log2(xlim) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$\ell$")
    plt.legend(loc=6)
    # f"axisymmetric_tiling_L{L}"
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(3)
    plt.close()


if __name__ == "__main__":
    main()
