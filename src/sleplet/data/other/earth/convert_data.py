from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

_file_location = Path(__file__).resolve()
_data_path = _file_location.parent


def _add_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    the tidal data doesn't have value for ell=0 or ell=1
    """
    df_missing = pd.DataFrame(
        [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
        columns=df.columns,
    )
    return df.append(df_missing, ignore_index=True)


def _helper(df: pd.DataFrame, filename: str) -> None:
    """
    method which gets the dataframe in the correct format and saves as mat file
    """
    # swap 'D' with 'E' in floats to convert to numeric
    df["real"] = pd.to_numeric(df["real"].str.replace("D", "E"))
    df["imag"] = pd.to_numeric(df["imag"].str.replace("D", "E"))

    # add missing tidal data
    df = df if (df["ell"] == 0).any() else _add_missing_data(df)

    # bandlimit
    L = df["ell"].max()

    # fill dataframe with zero for missing values
    # the original data doesn't have negative indices
    lst: list = []
    for ell in range(L + 1):
        lst.extend([ell, m, 0, 0] for m in range(-ell, 0))
    df_negatives = pd.DataFrame(np.array(lst), columns=df.columns.to_list())

    # combine dataframes and sort on ell & m
    df_concat = pd.concat([df, df_negatives])
    df_sorted = df_concat.sort_values(["ell", "m"]).reset_index(drop=True)

    # make complex flms
    df_sorted["flm"] = df_sorted["real"] + 1j * df_sorted["imag"]

    filename = f"EGM2008_{filename}_L{L}.mat"
    flm = np.array(df_sorted["flm"])
    mat = {"L": L, "flm": flm}
    sio.savemat(_data_path / filename, mat, oned_as="column")


def create_matfile(filename: str) -> None:
    """
    read original data file from
    Spherical Harmonic Coefficients for Earth's Elevation
    http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/first_release.html
    """
    # filename inputs and outputs
    inputs = [
        "Coeff_Height_and_Depth_to2190_DTM2006.0",
        "EGM2008_to2190_TideFree",
        "EGM2008_to2190_ZeroTide",
    ]
    outputs = ["Topography_flms", "tide_free", "zero_tide"]

    # read the data and drop error columns
    col_names = ["ell", "m", "real", "imag", "error_real", "error_imag"]
    df = pd.read_csv(_data_path / filename, delimiter="\\s+", names=col_names)
    df.drop(columns=["error_real", "error_imag"], inplace=True)

    output = outputs[inputs.index(filename)]
    _helper(df, output)


if __name__ == "__main__":
    filename = "Coeff_Height_and_Depth_to2190_DTM2006.0"
    # filename = "EGM2008_to2190_TideFree"
    # filename = "EGM2008_to2190_ZeroTide"
    create_matfile(filename)
