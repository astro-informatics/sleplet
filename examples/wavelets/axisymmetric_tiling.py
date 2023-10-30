import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

import sleplet

sns.set(context="paper")

B = 3
J_MIN = 2
L = 128
STEP = 0.01


def main() -> None:
    """Plot the tiling of the harmonic line."""
    xlim = L
    x = np.arange(xlim)
    xi = np.arange(0, xlim - 1 + STEP, STEP)
    kappas = sleplet.wavelet_methods.create_kappas(xlim, B, J_MIN)
    yi = scipy.interpolate.pchip(x, kappas[0])
    plt.semilogx(xi, yi(xi), label=r"$\Phi_{\ell0}$")
    for j, k in enumerate(kappas[1:]):
        yi = scipy.interpolate.pchip(x, k)
        plt.semilogx(xi, yi(xi), label=rf"$\Psi^{{{j+J_MIN}}}_{{\ell0}}$")
    plt.xlim(1, xlim)
    ticks = 2 ** np.arange(np.log2(xlim) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$\ell$")
    plt.legend(loc=6)
    print(f"Opening: axisymmetric_tiling_L{L}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


if __name__ == "__main__":
    main()
