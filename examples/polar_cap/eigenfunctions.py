from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import typing as npt

from sleplet import logger
from sleplet.data.setup_pooch import find_on_pooch_then_local
from sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from sleplet.utils.harmonic_methods import invert_flm_boosted
from sleplet.utils.plot_methods import calc_plot_resolution, save_plot

_fig_path = Path(__file__).resolve().parents[2] / "src" / "sleplet" / "figures"
sns.set(context="paper")

COLUMNS = 3
L = 16
ORDER = 0
PHI_IDX = 0
ROWS = 2
RESOLUTION = calc_plot_resolution(L)
SIGNS = [1, -1, 1, -1, 1, -1]
TEXT_BOX: dict[str, str | float] = {"boxstyle": "round", "color": "w"}
THETA_MAX = 40
THETA_MAX_DEFAULT = np.pi
THETA_MIN_DEFAULT = 0


def main() -> None:
    """
    create fig 5.1 from Spatiospectral Concentration on a Sphere by Simons et al 2006
    """
    x = np.linspace(THETA_MIN_DEFAULT, np.rad2deg(THETA_MAX_DEFAULT), RESOLUTION + 1)
    i = (x < THETA_MAX).sum()
    _, ax = plt.subplots(ROWS, COLUMNS, sharex="col", sharey="row")
    axes = ax.flatten()
    plt.setp(
        ax,
        xlim=[0, 180],
        xticks=[0, 40, 180],
        xticklabels=[r"$0^\circ$", r"$40^\circ$", r"$180^\circ$"],
        ylim=[-3, 3],
        yticks=[-2, 0, 2],
    )
    slepian = SlepianPolarCap(L, np.deg2rad(THETA_MAX), order=ORDER)
    for rank in range(ROWS * COLUMNS):
        _helper(axes[rank], slepian, x, i, rank)
    save_plot(_fig_path, "slepian_colatitude")


def _helper(
    ax: plt.Axes,
    slepian: SlepianPolarCap,
    x: npt.NDArray[np.float_],
    i: int,
    rank: int,
) -> None:
    """
    helper which plots the required order and specified ranks
    """
    logger.info(f"plotting rank={rank}")
    flm = slepian.eigenvectors[rank] * SIGNS[rank]
    lam = slepian.eigenvalues[rank]
    f = invert_flm_boosted(flm, L, RESOLUTION).real
    if rank > COLUMNS - 1:
        ax.set_xlabel(r"colatitude $\theta$")
    ax.plot(x[:i], f[:i, PHI_IDX], x[i:], f[i:, PHI_IDX])
    p = _find_p_value(rank, slepian.N)
    ax.text(
        0.39,
        0.92,
        rf"$\mu_{{{p+1}}}={{{lam:.6f}}}$",
        transform=ax.transAxes,
        bbox=TEXT_BOX,
    )


def _find_p_value(rank: int, shannon: int) -> int:
    """
    method to find the effective p rank of the Slepian function
    """
    orders = np.load(
        find_on_pooch_then_local(
            f"slepian_eigensolutions_D_polar{THETA_MAX}_L{L}_N{shannon}_orders.npy"
        )
    )
    return np.where(orders == ORDER)[0][rank]


if __name__ == "__main__":
    main()
