from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import pchip

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
    kappa0, kappa = s2let.axisym_wav_l(B, L, J_MIN)
    xi = np.arange(0, L - 1 + STEP, STEP)
    x = np.arange(L)
    yi = pchip(x, kappa0)
    plt.semilogx(xi, yi(xi))
    for k in kappa.T:
        yi = pchip(x, k)
        plt.semilogx(xi, yi(xi))
    plt.xlim([1, L])
    ticks = 2 ** np.arange(np.log2(L) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$\ell$")
    save_plot(fig_path, "tiling")


if __name__ == "__main__":
    main()
