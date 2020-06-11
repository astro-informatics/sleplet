import pandas as pd


def sort_and_clean_df(
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
