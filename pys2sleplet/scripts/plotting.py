#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from typing import Optional

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.plotting.create_plot import Plot
from pys2sleplet.utils.config import config
from pys2sleplet.utils.function_dicts import FUNCTIONS, MAPS
from pys2sleplet.utils.harmonic_methods import invert_flm_boosted
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.region import Region
from pys2sleplet.utils.string_methods import filename_angle


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
        "flm",
        type=valid_plotting,
        choices=list(FUNCTIONS.keys()),
        help="flm to plot on the sphere",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.75,
        help="alpha/phi pi fraction - defaults to 0",
    )
    parser.add_argument(
        "--annotation",
        "-n",
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
        "--region",
        "-r",
        action="store_true",
        help="flag which masks the function for a region (based on settings.toml)",
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
    args = parser.parse_args()
    return args


def plot(
    f: Functions,
    g: Optional[Functions] = None,
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
    filename = f"{f.name}{'' if f.region is None else f.region.name_ending}_L{f.L}_"
    multipole = f.multipole

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
        multipole = f.rotate(alpha_pi_fraction, beta_pi_fraction, gamma_pi_fraction)
    elif method == "translate":
        logger.info(
            f"angles: (alpha, beta) = ({alpha_pi_fraction}, {beta_pi_fraction})"
        )
        # don't add gamma if translation
        filename += f"{method}_{filename_angle(alpha_pi_fraction, beta_pi_fraction)}_"

        # translate by alpha, beta
        multipole = f.translate(alpha_pi_fraction, beta_pi_fraction)

    if g is not None:
        # perform convolution
        multipole = f.convolve(multipole, g.multipole)
        # adjust filename
        filename += f"convolved_{g.name}_L{f.L}_"

    # add resolution to filename
    filename += f"res{f.resolution}_"

    # create padded field to plot
    padded_field = invert_flm_boosted(multipole, f.L, f.resolution, reality=f.reality)

    # check for plotting type
    logger.info(f"plotting type: '{plot_type}'")
    if plot_type == "real":
        field = padded_field.real
    elif plot_type == "imag":
        field = padded_field.imag
    elif plot_type == "abs":
        field = np.abs(padded_field)
    elif plot_type == "sum":
        field = padded_field.real + padded_field.imag

    # turn off annotation if needed
    logger.info(f"annotations on: {annotations}")
    if annotations:
        annotation = f.annotations
    else:
        annotation = []

    # do plot
    filename += plot_type
    Plot(field, f.resolution, filename, annotation).execute()


def main() -> None:
    args = read_args()

    if args.region:
        mask = Region()
    else:
        mask = None

    f = FUNCTIONS[args.flm](config.L, extra_args=args.extra_args, region=mask)

    if args.convolve is not None:
        g = FUNCTIONS[args.convolve](f.L)
    else:
        g = None

    plot(
        f,
        g=g,
        method=args.method,
        plot_type=args.type,
        annotations=args.annotation,
        alpha_pi_fraction=args.alpha,
        beta_pi_fraction=args.beta,
        gamma_pi_fraction=args.gamma,
    )


if __name__ == "__main__":
    main()
