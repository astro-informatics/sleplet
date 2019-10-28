#!/usr/bin/env python
import os

import numpy as np
import pandas as pd
import scipy.io as sio


def create_matfile(filename: str) -> None:
    # read original data file from
    # Spherical Harmonic Coefficients for Earth's Elevation
    # http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/first_release.html
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    data = os.path.join(__location__, filename)

    # read in data
    col_names = ["ell", "m", "real", "imag"]
    df = pd.read_csv(data, delimiter="\s+", names=col_names)

    # swap 'D' with 'E' in floats to convert to numeric
    df["real"] = pd.to_numeric(df["real"].str.replace("D", "E"))
    df["imag"] = pd.to_numeric(df["imag"].str.replace("D", "E"))

    # band-limit
    L = df["ell"].values[-1]

    # fill dataframe with zero for missing values
    # the original data doesn't have negative indices
    lst = []
    for ell in range(L + 1):
        for m in range(-ell, 0):
            lst.append([ell, m, 0, 0])
    df_negatives = pd.DataFrame(np.array(lst), columns=col_names)

    # combine dataframes and sort on ell & m
    df_concat = pd.concat([df, df_negatives])
    df_sorted = df_concat.sort_values(["ell", "m"])

    # make complex flms
    df_sorted["flm"] = df_sorted["real"] + 1j * df_sorted["imag"]

    fname = f"EGM2008_Topography_flms_L{str(L)}.mat"
    filename = os.path.join(__location__, fname)
    flm = np.array(df_sorted["flm"])
    mat = {"L": L, "flm": flm}
    sio.savemat(filename, mat, oned_as="column")


if __name__ == "__main__":
    filename = "Coeff_Height_and_Depth_to2190_DTM2006.0"
    create_matfile(filename)
