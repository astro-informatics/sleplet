from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.interpolate import pchip

from pys2sleplet.flm.kernels.slepian_wavelets import SlepianWavelets
from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.utils.plot_methods import save_plot
from pys2sleplet.utils.pys2let import s2let
from pys2sleplet.utils.region import Region

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
    j_max = s2let.pys2let_j_max(B, L, J_MIN)
    j_vals = np.append(None, range(j_max - J_MIN + 1))
    x_lim = L ** 2
    x = range(x_lim)
    xi = np.arange(0, x_lim - 1 + STEP, STEP)
    region = Region(mask_name="south_america")
    for j in j_vals:
        sw = SlepianWavelets(L, B=B, j_min=J_MIN, j=j, region=region)
        sd = SlepianDecomposition(sw)
        k_p = sd.decompose_all()
        yi = pchip(x, k_p.real)
        label = r"$\Phi_p$" if j is None else rf"$\Psi^{j}_p$"
        plt.semilogx(xi, yi(xi), label=label)
    plt.axvline(x=sd.N, color="k")
    plt.annotate(
        f"N={sd.N}",
        xy=(sd.N + 2.5, 1.5),
        xytext=(0, 7),
        ha="center",
        textcoords="offset points",
        annotation_clip=False,
    )
    plt.xlim([1, x_lim])
    ticks = 2 ** np.arange(np.log2(x_lim) + 1, dtype=int)
    plt.xticks(ticks, ticks)
    plt.xlabel("p")
    plt.legend()
    save_plot(fig_path, f"slepian_tiling_south_america_L{L}")


if __name__ == "__main__":
    main()
