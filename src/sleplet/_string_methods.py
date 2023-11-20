import fractions
import re
import typing

import numpy as np

if typing.TYPE_CHECKING:
    import sleplet.functions.coefficients
    import sleplet.meshes.mesh_coefficients


def _get_angle_num_dem(angle_fraction: float) -> tuple[int, int]:
    """Get numerator and denominator for a given decimal."""
    angle = fractions.Fraction(angle_fraction).limit_denominator()
    return angle.numerator, angle.denominator


def _pi_in_filename(numerator: int, denominator: int) -> str:
    """Create filename for angle as multiple of pi."""
    filename = f"{numerator}pi"
    # if whole number
    if denominator != 1:
        filename += f"{denominator}"
    return filename


def filename_args(value: float, arg_name: str) -> str:
    """Add an extra argument to filename."""
    filename = "_"
    num, dem = _get_angle_num_dem(value)
    filename += f"{num}{arg_name}"
    if abs(value) < 1 and value != 0:
        filename += f"{dem}"
    return filename


def filename_angle(
    alpha_pi_fraction: float,
    beta_pi_fraction: float,
    gamma_pi_fraction: float = 0,
) -> str:
    """Middle part of filename."""
    # get numerator/denominator for filename
    alpha_num, alpha_den = _get_angle_num_dem(alpha_pi_fraction)
    beta_num, beta_den = _get_angle_num_dem(beta_pi_fraction)
    gamma_num, gamma_den = _get_angle_num_dem(gamma_pi_fraction)

    match (alpha_num, beta_num):
        case (0, 0):
            filename = "alpha0_beta0"
        case (0, _):
            filename = f"alpha0_beta{_pi_in_filename(beta_num, beta_den)}"
        case (_, 0):
            filename = f"alpha{_pi_in_filename(alpha_num, alpha_den)}_beta0"
        case _:
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
    Print the unicode pi with a prefix of the multiple unless it's 1
    i.e. pi, 2pi, 3pi.
    """
    multiple = int(angle / np.pi)
    return f"{multiple if multiple != 1 else ''}\u03C0"


def angle_as_degree(radian: float) -> int:
    """Convert radian angle to integer degree."""
    return round(np.rad2deg(radian))


def wavelet_ending(j_min: int, j: int | None) -> str:
    """Create the ending name of the given wavelet."""
    return "_scaling" if j is None else f"{filename_args(j + j_min, 'j')}"


def _convert_camel_case_to_snake_case(name: str) -> str:
    """Convert a string in camel case to snake case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def convert_classes_list_to_snake_case(
    classes: list[
        type["sleplet.functions.coefficients.Coefficients"]
        | type["sleplet.meshes.mesh_coefficients.MeshCoefficients"]
    ],
    *,
    word_to_remove: str = "",
) -> list[str]:
    """Convert a list of classes to snake case."""
    return [
        _convert_camel_case_to_snake_case(c.__name__.removeprefix(word_to_remove))
        for c in classes
    ]
