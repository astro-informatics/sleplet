from pathlib import Path

import seaborn as sns
from matplotlib import pyplot as plt

from pys2sleplet.plotting.polar_cap.inputs import ALPHA, SECOND_COLOUR, THETA_MAX, L
from pys2sleplet.plotting.polar_cap.utils import (
    create_table,
    earth_region_harmonic_coefficients,
    earth_region_slepian_coefficients,
)
from pys2sleplet.utils.config import settings

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set()


def main() -> None:
    """
    creates a plot of Slepian coefficient against rank
    """
    flm = earth_region_harmonic_coefficients(L, THETA_MAX)
    df = create_table(earth_region_slepian_coefficients, L, THETA_MAX)
    sns.scatterplot(x=range(len(flm)), y=flm, label="harmonic", linewidth=0, marker=".")
    sns.scatterplot(
        x=df.index,
        y=df["qty"],
        alpha=ALPHA,
        color=SECOND_COLOUR,
        label="slepian",
        linewidth=0,
        marker="*",
    )
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
