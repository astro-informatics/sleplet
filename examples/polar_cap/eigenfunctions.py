import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pooch
import seaborn as sns

import sleplet

sns.set(context="paper")

COLUMNS = 3
L = 16
ORDER = 0
PHI_IDX = 0
POOCH_RETRY = 3
ROWS = 2
RESOLUTION = sleplet.plot_methods.calc_plot_resolution(L)
SIGNS = [1, -1, 1, -1, 1, -1]
TEXT_BOX: dict[str, str | float] = {"boxstyle": "round", "color": "w"}
THETA_MAX = 40
THETA_MAX_DEFAULT = np.pi
THETA_MIN_DEFAULT = 0
ZENODO_DATA_DOI = "10.5281/zenodo.7767698"


def main() -> None:
    """
    Create fig 5.1 from Spatiospectral Concentration on a Sphere
    by Simons et al 2006.
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
    slepian = sleplet.slepian.SlepianPolarCap(L, np.deg2rad(THETA_MAX), order=ORDER)
    for rank in range(ROWS * COLUMNS):
        _helper(axes[rank], slepian, x, i, rank)
    print("Opening: slepian_colatitude")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


def _helper(
    ax: mpl.axes.Axes,
    slepian: sleplet.slepian.SlepianPolarCap,
    x: npt.NDArray[np.float64],
    i: int,
    rank: int,
) -> None:
    """Plot the required order and specified ranks."""
    print(f"plotting rank={rank}")
    flm = slepian.eigenvectors[rank] * SIGNS[rank]
    lam = slepian.eigenvalues[rank]
    f = sleplet.harmonic_methods.invert_flm_boosted(flm, L, RESOLUTION).real
    if rank > COLUMNS - 1:
        ax.set_xlabel(r"colatitude $\theta$")
    ax.plot(x[:i], f[:i, PHI_IDX], x[i:], f[i:, PHI_IDX])
    p = _find_p_value(rank, slepian.N)
    ax.text(
        0.39,
        0.92,
        rf"$\mu_{{{p + 1}}}={{{lam:.6f}}}$",
        transform=ax.transAxes,
        bbox=TEXT_BOX,
    )


def _find_p_value(rank: int, shannon: int) -> int:
    """Find the effective p rank of the Slepian function."""
    pooch_registry = pooch.create(
        path=pooch.os_cache("sleplet"),
        base_url=f"doi:{ZENODO_DATA_DOI}/",
        registry=None,
        retry_if_failed=POOCH_RETRY,
    )
    pooch_registry.load_registry_from_doi()
    orders = np.load(
        pooch_registry.fetch(
            f"slepian_eigensolutions_D_polar{THETA_MAX}_L{L}_N{shannon}_orders.npy",
            progressbar=True,
        ),
    )
    return np.where(orders == ORDER)[0][rank]


if __name__ == "__main__":
    main()
