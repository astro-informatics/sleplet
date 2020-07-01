from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle

from pys2sleplet.plotting.polar_cap.inputs import THETA_MAX, L
from pys2sleplet.plotting.polar_cap.utils import (
    create_table,
    earth_region_harmonic_coefficients,
    earth_region_slepian_coefficients,
)
from pys2sleplet.utils.config import settings

ORDERS = 15
RANKS = 100

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set()


def main() -> None:
    """
    creates a plot of Slepian coefficient against rank
    """
    df = create_table(
        earth_region_slepian_coefficients,
        L,
        THETA_MAX,
        ORDERS,
        RANKS,
        method="harmonic_sum",
    )
    sns.scatterplot(
        x=df.index,
        y=df["qty"],
        hue=df["m"],
        hue_order=df.sort_values(by="order")["m"],
        legend="full",
        style=df["m"],
        markers=MarkerStyle.filled_markers[:ORDERS],
    )
    flm = earth_region_harmonic_coefficients(L, THETA_MAX)[:RANKS]
    sns.scatterplot(
        x=range(len(flm)), y=flm, marker=".", color="black", label="harmonic"
    )
    plt.legend(ncol=3)
    plt.xlabel("coefficient")
    plt.ylabel("magnitude")

    plt.tight_layout()
    if settings.SAVE_FIG:
        for file_type in ["png", "pdf"]:
            filename = (
                fig_path / file_type / f"fp_earth_polar{THETA_MAX}_L{L}.{file_type}"
            )
            plt.savefig(filename, bbox_inches="tight")
    if settings.AUTO_OPEN:
        plt.show()


if __name__ == "__main__":
    main()
