import re
from fractions import Fraction
from typing import Any, Optional

import numpy as np


def _get_angle_num_dem(angle_fraction: float) -> tuple[int, int]:
    """
    ger numerator and denominator for a given decimal
    """
    angle = Fraction(angle_fraction).limit_denominator()
    return angle.numerator, angle.denominator


def _pi_in_filename(numerator: int, denominator: int) -> str:
    """
    create filename for angle as multiple of pi
    """
    filename = f"{numerator}pi"
    # if whole number
    if denominator != 1:
        filename += f"{denominator}"
    return filename


def filename_args(value: float, arg_name: str) -> str:
    """
    used to add an extra argument to filename
    """
    filename = "_"
    num, dem = _get_angle_num_dem(value)
    filename += f"{num}{arg_name}"
    if abs(value) < 1 and value != 0:
        filename += f"{dem}"
    return filename


def filename_angle(
    alpha_pi_fraction: float, beta_pi_fraction: float, gamma_pi_fraction: float = 0
) -> str:
    """
    middle part of filename
    """
    # get numerator/denominator for filename
    alpha_num, alpha_den = _get_angle_num_dem(alpha_pi_fraction)
    beta_num, beta_den = _get_angle_num_dem(beta_pi_fraction)
    gamma_num, gamma_den = _get_angle_num_dem(gamma_pi_fraction)

    # if alpha = beta = 0
    if not alpha_num and not beta_num:
        filename = "alpha0_beta0"
    # if alpha = 0
    elif not alpha_num:
        filename = f"alpha0_beta{_pi_in_filename(beta_num, beta_den)}"
    # if beta = 0
    elif not beta_num:
        filename = f"alpha{_pi_in_filename(alpha_num, alpha_den)}_beta0"
    # if alpha != 0 && beta !=0
    else:
        filename = (
            f"alpha{_pi_in_filename(alpha_num, alpha_den)}"
            f"_beta{_pi_in_filename(beta_num, beta_den)}"
        )

    # if rotation with gamma != 0
    if gamma_num:
        filename += f"_gamma{_pi_in_filename(gamma_num, gamma_den)}"
    return filename


def multiples_of_pi(angle: float) -> str:
    """
    prints the unicode pi with a prefix of the multiple unless it's 1
    i.e. pi, 2pi, 3pi
    """
    multiple = int(angle / np.pi)
    return f"{multiple if multiple != 1 else ''}\u03C0"


def angle_as_degree(radian: float) -> int:
    """
    converts radian angle to integer degree
    """
    return round(np.rad2deg(radian))


def wavelet_ending(j_min: int, j: Optional[int]) -> str:
    """
    the ending name of the given wavelet
    """
    return "_scaling" if j is None else f"{filename_args(j + j_min, 'j')}"


def convert_camel_case_to_snake_case(name: str) -> str:
    """
    converts a string in camel case to snake case
    """
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def convert_classes_list_to_snake_case(
    classes: list[Any], *, word_to_remove: str = ""
) -> list[str]:
    """
    converts a list of classes to snake case
    """
    return [
        convert_camel_case_to_snake_case(c.__name__.removeprefix(word_to_remove))
        for c in classes
    ]
