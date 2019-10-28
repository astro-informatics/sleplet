from fractions import Fraction
from typing import Tuple


def filename_args(angle: float, arg_name: str) -> str:
    """
    """
    filename = "_"
    num, dem = get_angle_num_dem(angle)
    filename += f"{num}{arg_name}"
    if angle < 1 and angle != 0:
        filename += f"{dem}"
    return filename


def get_angle_num_dem(angle_fraction: float) -> Tuple[int, int]:
    """
    ger numerator and denominator for a given decimal
    """
    angle = Fraction(angle_fraction).limit_denominator()
    return angle.numerator, angle.denominator


def missing_key(self, config: dict, key: str, value: str) -> None:
    """
    """
    try:
        setattr(self, key, config[key])
    except KeyError:
        setattr(self, key, value)


def pi_in_filename(numerator: int, denominator: int) -> str:
    """
    create filename for angle as multiple of pi
    """
    # if whole number
    if denominator == 1:
        filename = f"{numerator}pi"
    else:
        filename = f"{numerator}pi{denominator}"
    return filename
