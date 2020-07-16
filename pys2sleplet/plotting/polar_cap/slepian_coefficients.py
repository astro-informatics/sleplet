from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.plotting.inputs import TEXT_BOX, THETA_MAX
from pys2sleplet.plotting.polar_cap.utils import (
    create_table,
    earth_region_harmonic_coefficients,
    earth_region_slepian_coefficients,
    get_shannon,
)
from pys2sleplet.utils.plot_methods import save_plot

L = 19

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set(context="paper")


def main() -> None:
    """
    creates a plot of Slepian coefficient against rank
    """
    flm = earth_region_harmonic_coefficients(L, THETA_MAX)
    df = create_table(earth_region_slepian_coefficients, L, THETA_MAX)
    N = get_shannon(L, THETA_MAX)
    ax = plt.gca()
    sns.scatterplot(
        x=df.index, y=df["qty"], ax=ax, label="slepian", linewidth=0, marker="*"
    )
    sns.scatterplot(
        x=range(len(flm)), y=flm, ax=ax, label="harmonic", linewidth=0, marker="."
    )
    ax.axvline(x=N - 1, color="k")
    ax.text(0.17, 0.93, f"N={N}", transform=ax.transAxes, bbox=TEXT_BOX)
    ax.set_xlabel("coefficient")
    ax.set_ylabel("magnitude")
    save_plot(fig_path, f"fp_earth_polar{THETA_MAX}_L{L}")


if __name__ == "__main__":
    main()
