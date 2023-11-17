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
    """Plot the tiling of the Slepian line."""
    xlim = L**2
    x = np.arange(xlim)
    xi = np.arange(0, xlim - 1 + STEP, STEP)
    kappas = sleplet.wavelet_methods.create_kappas(xlim, B, J_MIN)
    yi = scipy.interpolate.pchip(x, kappas[0])
    plt.semilogx(xi, yi(xi), label=r"$\Phi_p$")
    for j, k in enumerate(kappas[1:]):
        yi = scipy.interpolate.pchip(x, k)
        plt.semilogx(xi, yi(xi), label=rf"$\Psi^{{{j+J_MIN}}}_p$")
    slepian = sleplet.slepian.SlepianArbitrary(L, "africa")
    plt.axvline(slepian.N, color="k", linestyle="dashed")
    plt.annotate(
        f"N={slepian.N}",
        xy=(slepian.N, 1),
        xytext=(19, 3),
        ha="center",
        textcoords="offset points",
        annotation_clip=False,
    )
    plt.xlim(1, xlim)
    ticks = list(2 ** np.arange(np.log2(xlim) + 1, dtype=int))
    plt.xticks(ticks, ticks)
    plt.xlabel(r"$p$")
    plt.legend()
    print(f"Opening: africa_slepian_tiling_L{L}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


if __name__ == "__main__":
    main()
