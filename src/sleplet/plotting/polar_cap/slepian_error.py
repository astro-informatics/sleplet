from pathlib import Path

import numpy as np
import pyssht as ssht
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import typing as npt
from sleplet.functions.flm.earth import Earth
from sleplet.plotting.inputs import THETA_MAX
from sleplet.slepian.slepian_functions import SlepianFunctions
from sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from sleplet.utils.plot_methods import save_plot
from sleplet.utils.region import Region
from sleplet.utils.slepian_methods import choose_slepian_method, slepian_forward
from sleplet.utils.vars import SAMPLING_SCHEME

L = 16

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")


def main() -> None:
    """
    creates a plot of Slepian coefficients against rank
    """
    region = Region(theta_max=np.deg2rad(THETA_MAX))
    earth = Earth(L, region=region)
    slepian = choose_slepian_method(L, region)
    field = ssht.inverse(earth.coefficients, L, Method=SAMPLING_SCHEME)
    integrate_region = _helper_region(
        L, slepian, field, earth.coefficients, slepian.mask
    )
    integrate_sphere = _helper_sphere(L, slepian, field, earth.coefficients)
    N = SlepianPolarCap(L, np.deg2rad(THETA_MAX)).N
    ax = plt.gca()
    sns.scatterplot(
        x=range(N), y=integrate_region, ax=ax, label="region", linewidth=0, marker="."
    )
    sns.scatterplot(
        x=range(N), y=integrate_sphere, ax=ax, label="sphere", linewidth=0, marker="*"
    )
    ax.set_xlabel("coefficients")
    ax.set_ylabel("relative error")
    ax.set_yscale("log")
    save_plot(fig_path, f"fp_error_earth_polar{THETA_MAX}_L{L}")


def _helper_sphere(
    L: int,
    slepian: SlepianFunctions,
    f: npt.NDArray[np.complex_],
    flm: npt.NDArray[np.complex_ | np.float_],
) -> npt.NDArray[np.float_]:
    """
    the difference in Slepian coefficients by integration of whole sphere
    """
    output = np.abs(slepian_forward(L, slepian, f=f))
    desired = np.abs(slepian_forward(L, slepian, flm=flm))
    return np.abs(output - desired) / desired


def _helper_region(
    L: int,
    slepian: SlepianFunctions,
    f: npt.NDArray[np.complex_],
    flm: npt.NDArray[np.complex_ | np.float_],
    mask: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    the difference in Slepian coefficients by integration of region on the sphere
    """
    output = np.abs(slepian_forward(L, slepian, f=f, mask=mask))
    desired = np.abs(slepian_forward(L, slepian, flm=flm))
    return np.abs(output - desired) / desired


if __name__ == "__main__":
    main()
