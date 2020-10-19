#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np
import pyssht as ssht
from pys2sleplet.figures.f_lm import F_LM

from pys2sleplet.functions.coefficients import Coefficients
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.function_dicts import FUNCTIONS, MAPS
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.plot_methods import calc_nearest_grid_point
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.slepian_methods import slepian_inverse
from pys2sleplet.utils.string_methods import filename_angle
from pys2sleplet.utils.vars import (
    ANNOTATION_SECOND_COLOUR,
    ARROW_STYLE,
    EARTH_ALPHA,
    EARTH_BETA,
    EARTH_GAMMA,
)


def valid_maps(map_name: str) -> str:
    """
    check if valid map
    """
    if map_name in MAPS:
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
        default=0.75,
        help="alpha/phi pi fraction - defaults to 0",
    )
    parser.add_argument(
        "--outline",
        "-o",
        action="store_false",
        help="flag which removes any annotation",
    )
    parser.add_argument(
        "--beta",
        "-b",
        type=float,
        default=0.125,
        help="beta/theta pi fraction - defaults to 0",
    )
    parser.add_argument(
        "--convolve",
        "-c",
        type=valid_maps,
        default=None,
        choices=list(MAPS.keys()),
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
        "--bandlimit", "-L", type=int, default=settings.L, help="bandlimit"
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
    parser.add_argument(
        "--noise", "-n", type=int, default=0, help="the SNR_IN of the noise level"
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
        default=0,
        help="the sigma of the applied smoothing",
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
    alpha_pi_fraction: Optional[float] = None,
    beta_pi_fraction: Optional[float] = None,
    gamma_pi_fraction: Optional[float] = None,
) -> None:
    """
    master plotting method
    """
    noised = "_noised" if f.noise else ""
    smoothed = "_smoothed" if f.smoothing else ""
    filename = f"{f.name}{noised}{smoothed}_L{f.L}_"
    coefficients = f.coefficients

    # turn off annotation if needed
    logger.info(f"annotations on: {annotations}")
    annotation = f.annotations if annotations else []

    # calculate angles
    alpha, beta = calc_nearest_grid_point(f.L, alpha_pi_fraction, beta_pi_fraction)
    gamma = gamma_pi_fraction * np.pi

    logger.info(f"plotting method: '{method}'")
    if method == "rotate":
        logger.info(
            "angles: (alpha, beta, gamma) = "
            f"({alpha_pi_fraction}, {beta_pi_fraction}, {gamma_pi_fraction})"
        )
        filename += (
            f"{method}_"
            f"{filename_angle(alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction)}_"
        )

        # rotate by alpha, beta, gamma
        coefficients = f.rotate(alpha, beta, gamma)
    elif method == "translate":
        logger.info(
            f"angles: (alpha, beta) = ({alpha_pi_fraction}, {beta_pi_fraction})"
        )
        # don't add gamma if translation
        filename += f"{method}_{filename_angle(alpha_pi_fraction, beta_pi_fraction)}_"

        # translate by alpha, beta
        coefficients = f.translate(alpha, beta)

        # annotate translation point
        x, y, z = ssht.s2_to_cart(beta, alpha)
        annotation.append(
            {**dict(x=x, y=y, z=z, arrowcolor=ANNOTATION_SECOND_COLOUR), **ARROW_STYLE}
        )

    if g is not None:
        # perform convolution
        coefficients = f.convolve(g.coefficients, coefficients)
        # adjust filename
        filename += f"convolved_{g.name}_L{f.L}_"

    # add resolution to filename
    if settings.UPSAMPLE:
        filename += f"res{f.resolution}_"

    # rotate plot of Earth to South America
    if f.__class__.__name__ == "Earth" or g.__class__.__name__ == "Earth":
        coefficients = ssht.rotate_flms(
            coefficients, EARTH_ALPHA, EARTH_BETA, EARTH_GAMMA, f.L
        )

    # get field value
    field = (
        ssht.inverse(coefficients, f.L, Reality=f.reality, Spin=f.spin)
        if isinstance(f, F_LM)
        else slepian_inverse(f.L, coefficients, f.slepian)
    )

    # do plot
    filename += plot_type
    Plot(
        field,
        f.L,
        f.resolution if settings.UPSAMPLE else f.L,
        filename,
        method="MWSS" if settings.UPSAMPLE else "MW",
        plot_type=plot_type,
        annotations=annotation,
        reality=f.reality,
        spin=f.spin,
    ).execute()


def main() -> None:
    args = read_args()

    mask = (
        Region(
            gap=settings.POLAR_GAP,
            mask_name=settings.SLEPIAN_MASK,
            phi_max=np.deg2rad(settings.PHI_MAX),
            phi_min=np.deg2rad(settings.PHI_MIN),
            theta_max=np.deg2rad(settings.THETA_MAX),
            theta_min=np.deg2rad(settings.THETA_MIN),
        )
        if args.region
        else None
    )

    f = FUNCTIONS[args.function](
        args.bandlimit,
        extra_args=args.extra_args,
        region=mask,
        noise=args.noise,
        smoothing=args.smoothing,
    )

    g = FUNCTIONS[args.convolve](args.bandlimit) if args.convolve is not None else None

    plot(
        f,
        g=g,
        method=args.method,
        plot_type=args.type,
        annotations=args.outline,
        alpha_pi_fraction=args.alpha,
        beta_pi_fraction=args.beta,
        gamma_pi_fraction=args.gamma,
    )


if __name__ == "__main__":
    main()
