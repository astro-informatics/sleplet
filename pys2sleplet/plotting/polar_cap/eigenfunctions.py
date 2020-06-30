from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.plotting.polar_cap.inputs import TEXT_BOX, THETA_MAX, L
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.harmonic_methods import invert_flm_boosted
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.vars import THETA_MAX_DEFAULT, THETA_MIN_DEFAULT

ORDERS = 5
PHI_IDX = 0
RANKS = 4
SIGNS = [[1, -1, 1, -1], [1, -1, -1, -1], [1, 1, 1, 1], [-1, -1, -1, -1], [1, 1, 1, 1]]


file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set()


def main() -> None:
    """
    create fig 5.1 from Spatiospectral Concentration on a Sphere by Simons et al 2006
    """
    resolution = calc_plot_resolution(L)
    x = np.linspace(THETA_MIN_DEFAULT, np.rad2deg(THETA_MAX_DEFAULT), resolution + 1)
    i = (x < THETA_MAX).sum() - 1
    _, ax = plt.subplots(ORDERS, RANKS, sharex="col", sharey="row")
    plt.setp(
        ax,
        xlim=[0, 180],
        xticks=[0, 40, 180],
        xticklabels=["$0^\circ$", "$40^\circ$", "$180^\circ$"],
        ylim=[-3, 3],
        yticks=[-2, 0, 2],
    )
    for order in range(ORDERS):
        slepian = SlepianPolarCap(L, np.deg2rad(THETA_MAX), order=order)
        for rank in range(RANKS):
            _helper(ax, slepian, resolution, x, i, order, rank)

    plt.tight_layout()
    if settings.SAVE_FIG:
        for file_type in ["png", "pdf"]:
            filename = fig_path / file_type / f"simons_5_1.{file_type}"
            plt.savefig(filename, bbox_inches="tight")
    if settings.AUTO_OPEN:
        plt.show()


def _helper(
    ax: np.ndarray,
    slepian: SlepianPolarCap,
    resolution: int,
    x: np.ndarray,
    i: int,
    order: int,
    rank: int,
) -> None:
    """
    helper which plots the required order and specified ranks
    """
    logger.info(f"plotting order={order}, rank={rank}")
    axs = ax[order, rank]
    flm = slepian.eigenvectors[rank] * SIGNS[order][rank]
    lam = slepian.eigenvalues[rank]
    f = invert_flm_boosted(flm, L, resolution).real
    if rank == 0:
        axs.set_ylabel(f"m = {order}")
    if order == 0:
        axs.set_title(fr"$\alpha$ = {rank}")
    if order == ORDERS - 1:
        axs.set_xlabel("colatitude $\Theta$")
    axs.plot(x[:i], f[:i, PHI_IDX], "b", x[i:], f[i:, PHI_IDX], "k")
    axs.text(
        0.33,
        0.75,
        fr"$\lambda$={lam:.6f}",
        transform=axs.transAxes,
        fontsize=8,
        bbox=TEXT_BOX,
    )


if __name__ == "__main__":
    main()
