from pathlib import Path

import numpy as np
import pyssht as ssht
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import pchip

from pys2sleplet.flm.kernels.slepian_wavelets import SlepianWavelets
from pys2sleplet.utils.plot_methods import save_plot
from pys2sleplet.utils.pys2let import s2let

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")

B = 2
J_MIN = 2
L = 128
STEP = 0.01


def main() -> None:
    """
    plots the tiling of the Slepian line
    """
    j_max = s2let.pys2let_j_max(B, L, J_MIN)
    j_vals = np.append(None, range(j_max - J_MIN + 1))
    x = range(L)
    xi = np.arange(0, L - 1 + STEP, STEP)
    for j in j_vals:
        sw = SlepianWavelets(L, B=B, j_min=J_MIN, j=j)
        kappa = [sw.multipole[i] for i in range(L ** 2) if ssht.ind2elm(i)[1] == 0]
        yi = pchip(x, kappa)
        plt.semilogx(xi, yi(xi))
    plt.xlim([1, L])
    ticks = 2 ** np.arange(np.log2(L) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$\ell$")
    save_plot(fig_path, "tiling")


if __name__ == "__main__":
    main()
