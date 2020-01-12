from fractions import Fraction
from typing import List, Tuple

from .vars import (
    PHI_MAX_DEFAULT,
    PHI_MIN_DEFAULT,
    SLEPIAN,
    THETA_MAX_DEFAULT,
    THETA_MIN_DEFAULT,
)


def get_angle_num_dem(angle_fraction: float) -> Tuple[int, int]:
    """
    ger numerator and denominator for a given decimal
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


def pi_in_filename(numerator: int, denominator: int) -> str:
    """
    create filename for angle as multiple of pi
    """
    filename = f"{numerator}pi"
    # if whole number
    if denominator != 1:
        filename += f"{denominator}"
    return filename


def filename_angle(
    alpha_pi_fraction: float, beta_pi_fraction: float, gamma_pi_fraction: float = 0
) -> str:
    """
    middle part of filename
    """
    # get numerator/denominator for filename
    alpha_num, alpha_den = get_angle_num_dem(alpha_pi_fraction)
    beta_num, beta_den = get_angle_num_dem(beta_pi_fraction)
    gamma_num, gamma_den = get_angle_num_dem(gamma_pi_fraction)

    # if alpha = beta = 0
    if not alpha_num and not beta_num:
        filename = "alpha0_beta0"
    # if alpha = 0
    elif not alpha_num:
        filename = f"alpha0_beta{pi_in_filename(beta_num, beta_den)}"
    # if beta = 0
    elif not beta_num:
        filename = f"alpha{pi_in_filename(alpha_num, alpha_den)}_beta0"
    # if alpha != 0 && beta !=0
    else:
        filename = (
            f"alpha{pi_in_filename(alpha_num, alpha_den)}"
            f"_beta{pi_in_filename(beta_num, beta_den)}"
        )

    # if rotation with gamma != 0
    if gamma_num:
        filename += f"_gamma{pi_in_filename(gamma_num, gamma_den)}"
    return filename


def filename_region():
    """
    middle part of filename for Slepian region
    """
    # initialise filename
    filename = ""

    # if phi min is not default
    if SLEPIAN["PHI_MIN"] != PHI_MIN_DEFAULT:
        filename += f"_pmin{SLEPIAN['PHI_MIN']}"

    # if phi max is not default
    if SLEPIAN["PHI_MAX"] != PHI_MAX_DEFAULT:
        filename += f"_pmax{SLEPIAN['PHI_MAX']}"

    # if theta min is not default
    if SLEPIAN["THETA_MIN"] != THETA_MIN_DEFAULT:
        filename += f"_tmin{SLEPIAN['THETA_MIN']}"

    # if theta max is not default
    if SLEPIAN["THETA_MAX"] != THETA_MAX_DEFAULT:
        filename += f"_tmax{SLEPIAN['THETA_MAX']}"
    return filename


def verify_args(args: List[int], num: int) -> None:
    if len(args) != num:
        raise ValueError(f"The number of extra arguments should be {num}")
