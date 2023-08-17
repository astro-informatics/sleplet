import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pyssht as ssht
import seaborn as sns

import sleplet

sns.set(context="paper")

L = 16
SAMPLING_SCHEME = "MWSS"
THETA_MAX = 40


def main() -> None:
    """Creates a plot of Slepian coefficients against rank."""
    region = sleplet.slepian.Region(theta_max=np.deg2rad(THETA_MAX))
    earth = sleplet.functions.Earth(L, region=region)
    field = ssht.inverse(earth.coefficients, L, Method=SAMPLING_SCHEME)
    integrate_region = _helper_region(L, region, field, earth.coefficients)
    integrate_sphere = _helper_sphere(L, region, field, earth.coefficients)
    N = sleplet.slepian.SlepianPolarCap(L, np.deg2rad(THETA_MAX)).N
    ax = plt.gca()
    sns.scatterplot(
        x=range(N),
        y=integrate_region,
        ax=ax,
        label="region",
        linewidth=0,
        marker=".",
    )
    sns.scatterplot(
        x=range(N),
        y=integrate_sphere,
        ax=ax,
        label="sphere",
        linewidth=0,
        marker="*",
    )
    ax.set_xlabel("coefficients")
    ax.set_ylabel("relative error")
    ax.set_yscale("log")
    print(f"Opening: fp_error_earth_polar{THETA_MAX}_L{L}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


def _helper_sphere(
    L: int,
    region: sleplet.slepian.Region,
    f: npt.NDArray[np.complex_],
    flm: npt.NDArray[np.complex_ | np.float_],
) -> npt.NDArray[np.float_]:
    """The difference in Slepian coefficients by integration of whole sphere."""
    slepian = sleplet.slepian_methods.choose_slepian_method(L, region)
    output = np.abs(sleplet.slepian_methods.slepian_forward(L, slepian, f=f))
    desired = np.abs(sleplet.slepian_methods.slepian_forward(L, slepian, flm=flm))
    return np.abs(output - desired) / desired


def _helper_region(
    L: int,
    region: sleplet.slepian.Region,
    f: npt.NDArray[np.complex_],
    flm: npt.NDArray[np.complex_ | np.float_],
) -> npt.NDArray[np.float_]:
    """The difference in Slepian coefficients by integration of region on the sphere."""
    slepian = sleplet.slepian_methods.choose_slepian_method(L, region)
    output = np.abs(
        sleplet.slepian_methods.slepian_forward(L, slepian, f=f, mask=slepian.mask),
    )
    desired = np.abs(sleplet.slepian_methods.slepian_forward(L, slepian, flm=flm))
    return np.abs(output - desired) / desired


if __name__ == "__main__":
    main()
