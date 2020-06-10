from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle

from pys2sleplet.plotting.polar_cap.inputs import TEXT_BOX, THETA_MAX, L
from pys2sleplet.slepian.slepian_region.slepian_polar_cap import SlepianPolarCap
from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger

ORDERS = 15
RANKS = 60

file_location = Path(__file__).resolve()
fig_path = file_location.parents[2] / "figures"
sns.set()


def main() -> None:
    """
    creates a plot of Slepian coefficient against rank
    """
    df = _create_eigenvalues()
    sns.scatterplot(
        x=df.index,
        y=df["lam"],
        hue=df["m"],
        hue_order=df.sort_values(by="order")["m"],
        legend="full",
        style=df["m"],
        markers=MarkerStyle.filled_markers[:ORDERS],
    )
    plt.xlabel("rank")
    plt.ylabel("$\lambda$")
    plt.text(0.05, 0.05, fr"$\Theta$={THETA_MAX}", fontsize=14, bbox=TEXT_BOX)
    if config.SAVE_FIG:
        for file_type in ["png", "pdf"]:
            filename = fig_path / file_type / f"simons_5-3.{file_type}"
            plt.savefig(filename)
    if config.AUTO_OPEN:
        plt.show()


def _create_eigenvalues() -> pd.DataFrame:
    """
    calculates all Slepian coefficients for all orders and sorts them
    """
    df = pd.DataFrame()
    for order in range(-(L - 1), L):
        logger.info(f"calculating order={order}")
        eigenvalues = _helper(order)
        df_tmp = pd.DataFrame()
        df_tmp["lam"] = eigenvalues
        df_tmp["order"] = abs(order)
        df = pd.concat([df, df_tmp], ignore_index=True)
    df_sorted = df.sort_values("lam", ascending=False).reset_index(drop=True)
    valid_orders = df_sorted["order"] < ORDERS
    df = df_sorted.loc[valid_orders].head(RANKS)
    df["m"] = df["order"].abs().astype(str)
    non_zero = df["order"] != 0
    df.loc[non_zero, "m"] = "$\pm$" + df.loc[non_zero, "m"]
    return df


def _helper(order: int) -> np.ndarray:
    """
    computes the Slepian coefficients for the given order
    """
    slepian = SlepianPolarCap(L, np.deg2rad(THETA_MAX), order=order)
    return slepian.eigenvalues


if __name__ == "__main__":
    main()
