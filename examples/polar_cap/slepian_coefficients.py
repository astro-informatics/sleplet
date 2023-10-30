import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns

import sleplet

sns.set(context="paper")

L = 16
THETA_MAX = 40


def _earth_region_harmonic_coefficients(
    L: int,
    theta_max: int,
) -> npt.NDArray[np.float_]:
    """Harmonic coefficients of the Earth for the polar cap region."""
    region = sleplet.slepian.Region(theta_max=np.deg2rad(theta_max))
    earth = sleplet.functions.Earth(L, region=region)
    coefficients = np.abs(earth.coefficients)
    coefficients[::-1].sort()
    return coefficients


def _earth_region_slepian_coefficients(
    L: int,
    theta_max: int,
) -> npt.NDArray[np.float_]:
    """Compute the Slepian coefficients."""
    region = sleplet.slepian.Region(theta_max=np.deg2rad(theta_max))
    earth = sleplet.functions.Earth(L, region=region)
    slepian = sleplet.slepian_methods.choose_slepian_method(L, region)
    return np.abs(
        sleplet.slepian_methods.slepian_forward(L, slepian, flm=earth.coefficients),
    )


def main() -> None:
    """Create a plot of Slepian coefficients against rank."""
    N = sleplet.slepian.SlepianPolarCap(L, np.deg2rad(THETA_MAX)).N
    flm = _earth_region_harmonic_coefficients(L, THETA_MAX)[:N]
    f_p = np.sort(_earth_region_slepian_coefficients(L, THETA_MAX))[::-1]
    ax = plt.gca()
    sns.scatterplot(x=range(N), y=f_p, ax=ax, label="slepian", linewidth=0, marker="*")
    sns.scatterplot(x=range(N), y=flm, ax=ax, label="harmonic", linewidth=0, marker=".")
    ax.set_xlabel("coefficients")
    ax.set_ylabel("magnitude")
    print(f"Opening: fp_earth_polar{THETA_MAX}_L{L}")
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(10)
    plt.close()


if __name__ == "__main__":
    main()
