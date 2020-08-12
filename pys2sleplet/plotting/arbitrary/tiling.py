from pathlib import Path

import numpy as np
import pyssht as ssht
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
L = 16
STEP = 0.01


def main() -> None:
    """
    plots the tiling of the Slepian line
    """
    kappas = _create_tiling_funcs()
    x_lim = L ** 2
    xi = np.arange(0, x_lim - 1 + STEP, STEP)
    x = np.arange(x_lim)
    for j, k in enumerate(kappas):
        label = r"$\Phi_p$" if j == 0 else rf"$\Psi^{j}_p$"
        yi = pchip(x, k)
        plt.semilogx(xi, yi(xi), label=label)
    plt.xlim([1, x_lim])
    ticks = 2 ** np.arange(np.log2(x_lim) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel("p")
    plt.legend()
    save_plot(fig_path, f"slepian_tiling_south_america_L{L}")


def _create_tiling_funcs() -> np.ndarray:
    """
    create tiling functions in Slepian space
    """
    kappa0, kappa = s2let.axisym_wav_l(B, L, J_MIN)
    kappas = np.zeros((kappa.shape[1] + 1, L ** 2))
    for ell in range(L):
        ind = ssht.elm2ind(ell, 0)
        kappas[0, ind] = kappa0[ell]
        for j in range(kappa.shape[1]):
            kappas[j + 1, ind] = kappa[ell, j]
    return kappas


if __name__ == "__main__":
    main()
