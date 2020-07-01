from typing import Callable, Optional

import numpy as np
import pandas as pd

from pys2sleplet.flm.maps.earth import Earth
from pys2sleplet.slepian.slepian_decomposition import SlepianDecomposition
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region


def earth_region_harmonic_coefficients(L: int, theta_max: int) -> np.ndarray:
    """
    harmonic coefficients of the Earth for the polar cap region
    """
    region = Region(theta_max=np.deg2rad(theta_max))
    earth = Earth(L, region=region)
    coefficients = np.abs(earth.multipole)
    coefficients[::-1].sort()
    return coefficients


def earth_region_slepian_coefficients(
    L: int, theta_max: int, order: int, method: str = "harmonic_sum"
) -> np.ndarray:
    """
    computes the Slepian coefficients for the given order
    """
    region = Region(theta_max=np.deg2rad(theta_max), order=order)
    earth = Earth(L, region=region)
    sd = SlepianDecomposition(earth)
    coefficients = np.abs(sd.decompose_all(method=method))
    return coefficients


def create_table(
    helper: Callable[..., float],
    L: int,
    theta_max: int,
    order_max: Optional[int] = None,
    rank_max: Optional[int] = None,
) -> pd.DataFrame:
    """
    calculates given quantity for all Slepian orders and sorts them
    """
    df = pd.DataFrame()
    for order in range(-(L - 1), L):
        logger.info(f"calculating order={order}")
        quantity = helper(L, theta_max, order)
        df_tmp = pd.DataFrame()
        df_tmp["qty"] = quantity
        df_tmp["order"] = abs(order)
        df = pd.concat([df, df_tmp], ignore_index=True)
    if order_max is None:
        order_max = L - 1
    if rank_max is None:
        rank_max = L * L
    df = _sort_and_clean_df(df, order_max, rank_max, "qty")
    return df


def _sort_and_clean_df(
    df: pd.DataFrame, order_max: int, rank_max: int, col: str
) -> pd.DataFrame:
    """
    helper method which prepares the incoming df for a scatter plot
    """
    df_sorted = df.sort_values(col, ascending=False).reset_index(drop=True)
    valid_orders = df_sorted["order"] < order_max
    df = df_sorted.loc[valid_orders].head(rank_max)
    df["m"] = df["order"].abs().astype(str)
    non_zero = df["order"] != 0
    df.loc[non_zero, "m"] = "$\pm$" + df.loc[non_zero, "m"]
    return df
