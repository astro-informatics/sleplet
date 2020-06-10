from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from pys2sleplet.plotting.polar_cap.inputs import ORDERS, RANKS, TEXT_BOX, THETA_MAX, L
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.config import config
from pys2sleplet.utils.harmonic_methods import invert_flm_boosted
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import calc_plot_resolution
from pys2sleplet.utils.vars import THETA_MAX_DEFAULT, THETA_MIN_DEFAULT

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"

resolution = calc_plot_resolution(L)
x = np.linspace(THETA_MIN_DEFAULT, np.rad2deg(THETA_MAX_DEFAULT), resolution + 1)
i = np.argwhere(x == np.rad2deg(THETA_MAX))[0][0] + 1


def main() -> None:
    """
    create fig 5.1 from Spatiospectral Concentration on a Sphere by Simons et al 2006
    """
    fig, ax = plt.subplots(ORDERS, RANKS, sharex="col", sharey="row")
    plt.setp(
        ax,
        xlim=[0, 180],
        xticks=[0, 40, 180],
        xticklabels=["$0^\circ$", "$40^\circ$", "$180^\circ$"],
        ylim=[-3, 3],
        yticks=[-2, 0, 2],
    )
    for order in range(ORDERS):
        slepian = SlepianPolarCap(L, THETA_MAX, order=order)
        for rank in range(RANKS):
            _helper(ax, slepian, order, rank)
    if config.SAVE_FIG:
        for file_type in ["png", "pdf"]:
            filename = fig_path / file_type / f"simons_5-1.{file_type}"
            plt.savefig(filename)
    if config.AUTO_OPEN:
        plt.show()


def _helper(ax: np.ndarray, slepian: SlepianPolarCap, order: int, rank: int) -> None:
    """
    helper which plots the required order and specified ranks
    """
    logger.info(f"plotting order={order}, rank={rank}")
    axs = ax[order, rank]
    flm = slepian.eigenvectors[rank]
    lam = slepian.eigenvalues[rank]
    f = invert_flm_boosted(flm, L, resolution).real
    if rank == 0:
        axs.set_ylabel(f"m = {order}")
    if order == 0:
        axs.set_title(fr"$\alpha$ = {rank}")
    if order == ORDERS - 1:
        axs.set_xlabel("colatitude $\Theta$")
    axs.plot(x[:i], f[:i, 0], "b", x[i:], f[i:, 0], "k")
    axs.text(
        0.3,
        0.75,
        fr"$\lambda$={lam:.6f}",
        transform=axs.transAxes,
        fontsize=8,
        bbox=TEXT_BOX,
    )
    axs.grid()


if __name__ == "__main__":
    main()
