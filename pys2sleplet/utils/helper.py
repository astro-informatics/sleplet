from fractions import Fraction
from typing import Tuple

import pyssht as ssht


def get_angle_num_dem(angle_fraction: float) -> Tuple[int, int]:
    """
    """
    angle = Fraction(angle_fraction).limit_denominator()
    return angle.numerator, angle.denominator


def filename_args(angle: float, arg_name: str) -> str:
    """
    """
    filename = "_"
    num, dem = get_angle_num_dem(angle)
    filename += f"{num}{arg_name}"
    if angle < 1 and angle != 0:
        filename += f"{dem}"
    return filename


def calc_samples(L: int) -> int:
    """
    calculate appropriate sample number for given L
    chosen such that have a two samples less than 0.1deg
    """
    if L == 1:
        samples = 1801
    elif L < 4:
        samples = 901
    elif L < 8:
        samples = 451
    elif L < 16:
        samples = 226
    elif L < 32:
        samples = 113
    elif L < 64:
        samples = 57
    elif L < 128:
        samples = 29
    elif L < 256:
        samples = 15
    elif L < 512:
        samples = 8
    elif L < 1024:
        samples = 4
    elif L < 2048:
        samples = 2
    else:
        samples = 1
    return samples


def ensure_f_bandlimited(grid_fun, L, reality):
    thetas, phis = ssht.sample_positions(L, Grid=True, Method="MWSS")
    f = grid_fun(thetas, phis)
    flm = ssht.forward(f, L, Reality=reality, Method="MWSS")
    return flm
