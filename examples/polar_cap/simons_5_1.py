import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import typing as npt

from sleplet.harmonic_methods import invert_flm_boosted
from sleplet.plot_methods import calc_plot_resolution, save_plot
from sleplet.slepian import SlepianPolarCap

sns.set(context="paper")

L = 16
ORDERS = 4
PHI_IDX = 0
RANKS = 3
RESOLUTION = calc_plot_resolution(L)
SIGNS = [[1, -1, 1], [-1, -1, -1], [1, 1, 1], [1, 1, -1]]
TEXT_BOX: dict[str, str | float] = {"boxstyle": "round", "color": "w"}
THETA_MAX = 40
THETA_MAX_DEFAULT = np.pi
THETA_MIN_DEFAULT = 0


def main() -> None:
    """
    Create fig 5.1 from Spatiospectral Concentration on a Sphere
    by Simons et al 2006.
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
    save_plot("slepian_colatitude")


def _helper(  # noqa: PLR0913
    ax: npt.NDArray,
    slepian: SlepianPolarCap,
    resolution: int,
    x: npt.NDArray[np.float_],
    i: int,
    order: int,
    rank: int,
) -> None:
    """Helper which plots the required order and specified ranks."""
    print(f"plotting order={order}, rank={rank}")
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
        0.45,
        0.81,
        rf"$\mu={{{lam:.6f}}}$",
        transform=axs.transAxes,
        bbox=TEXT_BOX,
    )


if __name__ == "__main__":
    main()
