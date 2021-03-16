#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.functions.coefficients import Coefficients
from pys2sleplet.functions.f_lm import F_LM
from pys2sleplet.plotting.create_plot_sphere import Plot
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.function_dicts import FUNCTIONS, MAPS_LM
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.mask_methods import create_default_region
from pys2sleplet.utils.plot_methods import (
    calc_nearest_grid_point,
    rotate_earth_to_south_america,
)
from pys2sleplet.utils.slepian_methods import slepian_forward, slepian_inverse
from pys2sleplet.utils.string_methods import filename_angle, filename_args
from pys2sleplet.utils.vars import (
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
    if map_name in MAPS_LM:
        function = map_name
    else:
        raise ValueError("Not a valid map name to convolve")
    return function


def valid_plotting(func_name: str) -> str:
    """
    check if valid function
    """
    if func_name in FUNCTIONS:
        function = func_name
    else:
        raise ValueError("Not a valid function name to plot")
    return function


def read_args() -> Namespace:
    """
    method to read args from the command line
    """
    parser = ArgumentParser(description="Create SSHT plot")
    parser.add_argument(
        "function",
        type=valid_plotting,
        choices=list(FUNCTIONS.keys()),
        help="function to plot on the sphere",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=ALPHA_DEFAULT,
        help="alpha/phi pi fraction - defaults to 0",
    )
    parser.add_argument(
        "--bandlimit", "-L", type=int, default=settings.L, help="bandlimit"
    )
    parser.add_argument(
        "--beta",
        "-b",
        type=float,
        default=BETA_DEFAULT,
        help="beta/theta pi fraction - defaults to 0",
    )
    parser.add_argument(
        "--convolve",
        "-c",
        type=valid_maps,
        default=None,
        choices=list(MAPS_LM.keys()),
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
    return parser.parse_args()


def plot(
    f: Coefficients,
    g: Optional[Coefficients] = None,
    method: str = "north",
    plot_type: str = "real",
    annotations: bool = True,
    alpha_pi_frac: Optional[float] = None,
    beta_pi_frac: Optional[float] = None,
    gamma_pi_frac: Optional[float] = None,
) -> None:
    """
    master plotting method
    """
    noised = f"{filename_args(f.noise, 'noise')}" if f.noise is not None else ""
    smoothing = (
        f"{filename_args(f.smoothing, 'smoothed')}" if f.smoothing is not None else ""
    )
    filename = f"{f.name}{noised}{smoothing}_L{f.L}"
    coefficients = f.coefficients

    # turn off annotation if needed
    logger.info(f"annotations on: {annotations}")
    annotation = []

    # Shannon number for Slepian coefficients
    shannon = f.slepian.N if not isinstance(f, F_LM) else None

    logger.info(f"plotting method: '{method}'")
    if method == "rotate":
        coefficients, filename = rotation_helper(
            f, filename, alpha_pi_frac, beta_pi_frac, gamma_pi_frac
        )
    elif method == "translate":
        coefficients, filename, trans_annotation = translation_helper(
            f, filename, alpha_pi_frac, beta_pi_frac, shannon
        )

        # annotate translation point
        if annotations:
            annotation.append(trans_annotation)

    if isinstance(g, Coefficients):
        coefficients, filename = convolution_helper(
            f, g, coefficients, shannon, filename
        )

    # rotate plot of Earth to South America
    if "earth" in filename:
        coefficients = rotate_earth_to_south_america(coefficients, f.L)

    # get field value
    field = (
        ssht.inverse(
            coefficients, f.L, Reality=f.reality, Spin=f.spin, Method=SAMPLING_SCHEME
        )
        if isinstance(f, F_LM)
        else slepian_inverse(coefficients, f.L, f.slepian)
    )

    # do plot
    Plot(
        field,
        f.L,
        filename,
        annotations=annotation,
        plot_type=plot_type,
        reality=f.reality,
        region=f.region if not isinstance(f, F_LM) else None,
        spin=f.spin,
    ).execute()


def main() -> None:
    args = read_args()

    mask = create_default_region(settings) if args.region else None

    f = FUNCTIONS[args.function](
        args.bandlimit,
        extra_args=args.extra_args,
        region=mask,
        noise=args.noise if args.noise is not None else None,
        smoothing=args.smoothing if args.smoothing is not None else None,
    )

    g = (
        FUNCTIONS[args.convolve](args.bandlimit)
        if isinstance(args.convolve, str)
        else None
    )

    plot(
        f,
        g=g,
        method=args.method,
        plot_type=args.type,
        annotations=args.outline,
        alpha_pi_frac=args.alpha,
        beta_pi_frac=args.beta,
        gamma_pi_frac=args.gamma,
    )


def rotation_helper(
    f: Coefficients,
    filename: str,
    alpha_pi_frac: Optional[float],
    beta_pi_frac: Optional[float],
    gamma_pi_frac: Optional[float],
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
    coefficients = f.rotate(alpha, beta, gamma)
    return coefficients, filename


def translation_helper(
    f: Coefficients,
    filename: str,
    alpha_pi_frac: Optional[float],
    beta_pi_frac: Optional[float],
    shannon: int,
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


def convolution_helper(
    f: Coefficients,
    g: Coefficients,
    coefficients: np.ndarray,
    shannon: int,
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

    filename += f"_convolved_{g.name}_L{f.L}"
    return coefficients, filename


if __name__ == "__main__":
    main()
