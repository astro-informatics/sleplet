from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import typing as npt

from sleplet.plotting.inputs import TEXT_BOX, THETA_MAX
from sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from sleplet.utils.harmonic_methods import invert_flm_boosted
from sleplet.utils.logger import logger
from sleplet.utils.plot_methods import calc_plot_resolution, save_plot
from sleplet.utils.vars import THETA_MAX_DEFAULT, THETA_MIN_DEFAULT

L = 16
ORDERS = 4
PHI_IDX = 0
RANKS = 3
RESOLUTION = calc_plot_resolution(L)
SIGNS = [[1, -1, 1], [-1, -1, -1], [1, 1, 1], [1, 1, -1]]


file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")


def main() -> None:
    """
    create fig 5.1 from Spatiospectral Concentration on a Sphere by Simons et al 2006
    """
    x = np.linspace(THETA_MIN_DEFAULT, np.rad2deg(THETA_MAX_DEFAULT), RESOLUTION + 1)
    i = (x < THETA_MAX).sum()
    _, ax = plt.subplots(ORDERS, RANKS, sharex="col", sharey="row")
    plt.setp(
        ax,
        xlim=[0, 180],
        xticks=[0, 40, 180],
        xticklabels=[r"$0^\circ$", r"$40^\circ$", r"$180^\circ$"],
        ylim=[-3, 3],
        yticks=[-2, 0, 2],
    )
    for order in range(ORDERS):
        slepian = SlepianPolarCap(L, np.deg2rad(THETA_MAX), order=order)
        for rank in range(RANKS):
            _helper(ax, slepian, RESOLUTION, x, i, order, rank)
    save_plot(fig_path, "slepian_colatitude")


def _helper(
    ax: npt.NDArray,
    slepian: SlepianPolarCap,
    resolution: int,
    x: npt.NDArray,
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
        axs.set_ylabel(rf"$m={{{order}}}$")
    if order == 0:
        axs.set_title(rf"$\alpha={{{rank+1}}}$")
    if order == ORDERS - 1:
        axs.set_xlabel("colatitude")
    axs.plot(x[:i], f[:i, PHI_IDX], x[i:], f[i:, PHI_IDX])
    axs.text(
        0.45, 0.81, rf"$\mu={{{lam:.6f}}}$", transform=axs.transAxes, bbox=TEXT_BOX
    )


if __name__ == "__main__":
    main()
