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
J_MIN = 0
L = 16
STEP = 0.01


def main() -> None:
    """
    plots the tiling of the Slepian line
    """
    xlim = L ** 2
    x = np.arange(xlim)
    xi = np.arange(0, xlim - 1 + STEP, STEP)
    kappa0, kappa = s2let.axisym_wav_l(B, xlim, J_MIN)
    yi = pchip(x, kappa0)
    plt.semilogx(xi, yi(xi), label=r"$\Phi_p$")
    for j, k in enumerate(kappa.T):
        yi = pchip(x, k)
        plt.semilogx(xi, yi(xi), label=rf"$\Psi^{j+J_MIN}_p$")
    plt.xlim([1, xlim])
    ticks = 2 ** np.arange(np.log2(xlim) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel("p")
    plt.legend()
    save_plot(fig_path, f"slepian_tiling_L{L}")


if __name__ == "__main__":
    main()
