#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np
import pyssht as ssht

from sleplet.functions.coefficients import Coefficients
from sleplet.functions.f_lm import F_LM
from sleplet.plotting.create_plot_sphere import Plot
from sleplet.utils.class_lists import COEFFICIENTS, MAPS_LM
from sleplet.utils.config import settings
from sleplet.utils.logger import logger
from sleplet.utils.mask_methods import create_default_region
from sleplet.utils.plot_methods import (
    calc_nearest_grid_point,
    rotate_earth_to_africa,
    rotate_earth_to_south_america,
)
from sleplet.utils.slepian_methods import slepian_forward, slepian_inverse
from sleplet.utils.string_methods import (
    convert_classes_list_to_snake_case,
    filename_angle,
)
from sleplet.utils.vars import (
    ALPHA_DEFAULT,
    ANNOTATION_COLOUR,
    ARROW_STYLE,
    BETA_DEFAULT,
    SAMPLING_SCHEME,
)


def valid_maps(map_name: str) -> str:
    """
    check if valid map
    """
    if map_name in convert_classes_list_to_snake_case(MAPS_LM):
        return map_name
    else:
        raise ValueError(f"{map_name} is not a valid map to convolve")


def valid_plotting(func_name: str) -> str:
    """
    check if valid function
    """
    if func_name in convert_classes_list_to_snake_case(COEFFICIENTS):
        return func_name
    else:
        raise ValueError(f"{func_name} is not a valid function to plot")


def read_args() -> Namespace:
    """
    method to read args from the command line
    """
    parser = ArgumentParser(description="Create SSHT plot")
    parser.add_argument(
        "function",
        type=valid_plotting,
        choices=convert_classes_list_to_snake_case(COEFFICIENTS),
        help="function to plot on the sphere",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=ALPHA_DEFAULT,
        help=f"alpha/phi pi fraction - defaults to {ALPHA_DEFAULT}",
    )
    parser.add_argument(
        "--bandlimit", "-L", type=int, default=settings.L, help="bandlimit"
    )
    parser.add_argument(
        "--beta",
        "-b",
        type=float,
        default=BETA_DEFAULT,
        help=f"beta/theta pi fraction - defaults to {BETA_DEFAULT}",
    )
    parser.add_argument(
        "--convolve",
        "-c",
        type=valid_maps,
        default=None,
        choices=convert_classes_list_to_snake_case(MAPS_LM),
        help="glm to perform sifting convolution with i.e. flm x glm*",
    )
    parser.add_argument(
        "--extra_args",
        "-e",
        type=int,
        nargs="+",
        help="list of extra args for functions",
    )
    parser.add_argument(
        "--gamma",
        "-g",
        type=float,
        default=0,
        help="gamma pi fraction - defaults to 0 - rotation only",
    )
    parser.add_argument(
        "--method",
        "-m",
        type=str,
        nargs="?",
        default="north",
        const="north",
        choices=["north", "rotate", "translate"],
        help="plotting routine: defaults to north",
    )
    parser.add_argument("--noise", "-n", type=int, help="the SNR_IN of the noise level")
    parser.add_argument(
        "--outline",
        "-o",
        action="store_false",
        help="flag which removes any annotation",
    )
    parser.add_argument(
        "--region",
        "-r",
        action="store_true",
        help="flag which masks the function for a region (based on settings.toml)",
    )
    parser.add_argument(
        "--smoothing",
        "-s",
        type=int,
        help="the scaling of the sigma in Gaussian smoothing of the Earth",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        nargs="?",
        default="real",
        const="real",
        choices=["abs", "real", "imag", "sum"],
        help="plotting type: defaults to real",
    )
    parser.add_argument(
        "--unnormalise",
        "-u",
        action="store_true",
        help="flag turns off normalisation for plot",
    )
    parser.add_argument(
        "--unzeropad",
        "-z",
        action="store_true",
        help="flag turns off upsampling for plot",
    )
    parser.add_argument(
        "--view",
        "-v",
        type=str,
        nargs="?",
        default="south_america",
        const="south_america",
        choices=["africa", "south_america"],
        help="view of Earth: defaults to 'south_america'",
    )
    return parser.parse_args()


def plot(
    f: Coefficients,
    g: Optional[Coefficients],
    alpha_pi_frac: float,
    beta_pi_frac: float,
    gamma_pi_frac: float,
    annotations: bool,
    normalise: bool,
    method: str,
    plot_type: str,
    upsample: bool,
    earth_view: str,
    amplitude: Optional[float],
) -> None:
    """
    master plotting method
    """
    filename = f.name
    coefficients = f.coefficients

    # turn off annotation if needed
    logger.info(f"annotations on: {annotations}")
    annotation = []

    # Shannon number for Slepian coefficients
    shannon = None if isinstance(f, F_LM) else f.slepian.N

    logger.info(f"plotting method: '{method}'")
    if method == "rotate":
        coefficients, filename = _rotation_helper(
            f, filename, alpha_pi_frac, beta_pi_frac, gamma_pi_frac
        )
    elif method == "translate":
        coefficients, filename, trans_annotation = _translation_helper(
            f, filename, alpha_pi_frac, beta_pi_frac, shannon
        )

        # annotate translation point
        if annotations:
            annotation.append(trans_annotation)

    if isinstance(g, Coefficients):
        coefficients, filename = _convolution_helper(
            f, g, coefficients, shannon, filename
        )

    # rotate plot of Earth
    if "earth" in filename:
        if earth_view == "africa":
            coefficients = rotate_earth_to_africa(coefficients, f.L)
            filename += "_africa"
        elif earth_view == "south_america":
            coefficients = rotate_earth_to_south_america(coefficients, f.L)

    # get field value
    field = _coefficients_to_field(f, coefficients)

    # do plot
    Plot(
        field,
        f.L,
        filename,
        amplitude=amplitude,
        annotations=annotation,
        normalise=normalise,
        plot_type=plot_type,
        reality=f.reality,
        region=None if isinstance(f, F_LM) else f.region,
        spin=f.spin,
        upsample=upsample,
    ).execute()


def _rotation_helper(
    f: Coefficients,
    filename: str,
    alpha_pi_frac: float,
    beta_pi_frac: float,
    gamma_pi_frac: float,
) -> tuple[np.ndarray, str]:
    """
    performs the rotation specific steps
    """
    logger.info(
        "angles: (alpha, beta, gamma) = "
        f"({alpha_pi_frac}, {beta_pi_frac}, {gamma_pi_frac})"
    )
    filename += f"_rotate_{filename_angle(alpha_pi_frac, beta_pi_frac, gamma_pi_frac)}"

    # calculate angles
    alpha, beta = calc_nearest_grid_point(f.L, alpha_pi_frac, beta_pi_frac)
    gamma = gamma_pi_frac * np.pi

    # rotate by alpha, beta, gamma
    coefficients = f.rotate(alpha, beta, gamma=gamma)
    return coefficients, filename


def _translation_helper(
    f: Coefficients,
    filename: str,
    alpha_pi_frac: float,
    beta_pi_frac: float,
    shannon: Optional[int],
) -> tuple[np.ndarray, str, dict]:
    """
    performs the translation specific steps
    """
    logger.info(f"angles: (alpha, beta) = ({alpha_pi_frac}, {beta_pi_frac})")
    # don't add gamma if translation
    filename += f"_translate_{filename_angle(alpha_pi_frac, beta_pi_frac)}"

    # calculate angles
    alpha, beta = calc_nearest_grid_point(f.L, alpha_pi_frac, beta_pi_frac)

    # translate by alpha, beta
    coefficients = f.translate(alpha, beta, shannon=shannon)

    # annotate translation point
    x, y, z = ssht.s2_to_cart(beta, alpha)
    annotation = {
        **dict(x=x, y=y, z=z, arrowcolor=ANNOTATION_COLOUR),
        **ARROW_STYLE,
    }
    return coefficients, filename, annotation


def _convolution_helper(
    f: Coefficients,
    g: Coefficients,
    coefficients: np.ndarray,
    shannon: Optional[int],
    filename: str,
) -> tuple[np.ndarray, str]:
    """
    performs the convolution specific steps
    """
    g_coefficients = (
        g.coefficients
        if isinstance(f, F_LM)
        else slepian_forward(f.L, f.slepian, flm=g.coefficients)
    )
    coefficients = f.convolve(g_coefficients, coefficients, shannon=shannon)

    filename += f"_convolved_{g.name}"
    return coefficients, filename


def _coefficients_to_field(f: Coefficients, coefficients: np.ndarray) -> np.ndarray:
    """
    computes the field over the samples from the harmonic/Slepian coefficients
    """
    return (
        ssht.inverse(
            coefficients, f.L, Reality=f.reality, Spin=f.spin, Method=SAMPLING_SCHEME
        )
        if isinstance(f, F_LM)
        else slepian_inverse(coefficients, f.L, f.slepian)
    )


def _compute_amplitude_for_noisy_plots(f: Coefficients) -> Optional[float]:
    """
    for the noised plots fix the amplitude to the initial data
    """
    return (
        np.abs(_coefficients_to_field(f, f.unnoised_coefficients)).max()
        if f.noise is not None
        else None
    )


def main() -> None:
    args = read_args()

    mask = create_default_region(settings) if args.region else None

    f = COEFFICIENTS[
        convert_classes_list_to_snake_case(COEFFICIENTS).index(args.function)
    ](
        args.bandlimit,
        extra_args=args.extra_args,
        region=mask,
        noise=args.noise if args.noise is not None else None,
        smoothing=args.smoothing if args.smoothing is not None else None,
    )

    g = (
        MAPS_LM[convert_classes_list_to_snake_case(MAPS_LM).index(args.convolve)](
            args.bandlimit
        )
        if isinstance(args.convolve, str)
        else None
    )

    # custom amplitude for noisy plots
    amplitude = _compute_amplitude_for_noisy_plots(f)

    plot(
        f,
        g,
        args.alpha,
        args.beta,
        args.gamma,
        args.outline,
        not args.unnormalise,
        args.method,
        args.type,
        not args.unzeropad,
        args.view,
        amplitude,
    )


if __name__ == "__main__":
    main()
