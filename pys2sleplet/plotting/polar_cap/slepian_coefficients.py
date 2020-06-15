from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle

from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.plotting.polar_cap.inputs import THETA_MAX, L
from pys2sleplet.plotting.polar_cap.utils import sort_and_clean_df
from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region

ORDERS = 15
RANKS = 100

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set()


def main() -> None:
    """
    creates a plot of Slepian coefficient against rank
    """
    df = _create_coefficients()
    sns.scatterplot(
        x=df.index,
        y=df["f_p"],
        hue=df["m"],
        hue_order=df.sort_values(by="order")["m"],
        legend="full",
        style=df["m"],
        markers=MarkerStyle.filled_markers[:ORDERS],
    )
    flm = _harmonic_coefficients()
    sns.scatterplot(x=range(RANKS), y=flm, marker="+", color="black", label="harmonic")
    plt.legend(ncol=3)
    plt.xlabel("coefficient")
    plt.ylabel("magnitude")

    plt.tight_layout()
    if config.SAVE_FIG:
        for file_type in ["png", "pdf"]:
            filename = fig_path / file_type / f"fp_polar{THETA_MAX}_L{L}.{file_type}"
            plt.savefig(filename, bbox_inches="tight")
    if config.AUTO_OPEN:
        plt.show()


def _create_coefficients() -> pd.DataFrame:
    """
    calculates all Slepian coefficients for all orders and sorts them
    """
    df = pd.DataFrame()
    for order in range(-(L - 1), L):
        logger.info(f"calculating order={order}")
        coefficients = np.abs(_helper(order))
        df_tmp = pd.DataFrame()
        df_tmp["f_p"] = coefficients
        df_tmp["order"] = abs(order)
        df = pd.concat([df, df_tmp], ignore_index=True)
    df = sort_and_clean_df(df, ORDERS, RANKS, "f_p")
    return df


def _helper(order: int) -> np.ndarray:
    """
    computes the Slepian coefficients for the given order
    """
    region = Region(theta_max=np.deg2rad(THETA_MAX), order=order)
    earth = Earth(L, region=region)
    sd = SlepianDecomposition(earth)
    coefficients = sd.decompose_all()
    return coefficients


def _harmonic_coefficients() -> np.ndarray:
    """
    harmonic coefficients of the Earth for the polar cap region
    """
    region = Region(theta_max=np.deg2rad(THETA_MAX))
    earth = Earth(L, region=region)
    coefficients = np.abs(earth.multipole[:RANKS])
    return coefficients


if __name__ == "__main__":
    main()
