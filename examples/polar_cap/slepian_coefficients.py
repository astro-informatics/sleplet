import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import typing as npt

from sleplet.functions import Earth
from sleplet.plot_methods import save_plot
from sleplet.slepian import Region, SlepianPolarCap
from sleplet.slepian_methods import choose_slepian_method, slepian_forward

sns.set(context="paper")

L = 16
THETA_MAX = 40


def _earth_region_harmonic_coefficients(
    L: int,
    theta_max: int,
) -> npt.NDArray[np.float_]:
    """Harmonic coefficients of the Earth for the polar cap region."""
    region = Region(theta_max=np.deg2rad(theta_max))
    earth = Earth(L, region=region)
    coefficients = np.abs(earth.coefficients)
    coefficients[::-1].sort()
    return coefficients


def _earth_region_slepian_coefficients(
    L: int,
    theta_max: int,
) -> npt.NDArray[np.float_]:
    """Computes the Slepian coefficients."""
    region = Region(theta_max=np.deg2rad(theta_max))
    earth = Earth(L, region=region)
    slepian = choose_slepian_method(L, region)
    return np.abs(slepian_forward(L, slepian, flm=earth.coefficients))


def main() -> None:
    """Creates a plot of Slepian coefficients against rank."""
    N = SlepianPolarCap(L, np.deg2rad(THETA_MAX)).N
    flm = _earth_region_harmonic_coefficients(L, THETA_MAX)[:N]
    f_p = np.sort(_earth_region_slepian_coefficients(L, THETA_MAX))[::-1]
    ax = plt.gca()
    sns.scatterplot(x=range(N), y=f_p, ax=ax, label="slepian", linewidth=0, marker="*")
    sns.scatterplot(x=range(N), y=flm, ax=ax, label="harmonic", linewidth=0, marker=".")
    ax.set_xlabel("coefficients")
    ax.set_ylabel("magnitude")
    save_plot(f"fp_earth_polar{THETA_MAX}_L{L}")


if __name__ == "__main__":
    main()
