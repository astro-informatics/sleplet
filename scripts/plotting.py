#!/usr/bin/env python
from argparse import ArgumentParser, Namespace
from configparser import ConfigParser

from pys2sleplet.flm.functions import functions
from pys2sleplet.sifting_convolution import SiftingConvolution


def read_args() -> Namespace:
    """
    """
    parser = ArgumentParser(description="Create SSHT plot")
    parser.add_argument(
        "flm",
        type=valid_plotting,
        choices=list(functions().keys()),
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
        help="flag which if passed removes any annotation",
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
        type=valid_kernels,
        choices=list(functions().keys()),
        help="glm to perform sifting convolution with i.e. flm x glm*",
    )
    parser.add_argument(
        "--double",
        "-d",
        action="store_true",
        help="flag which if passed creates a double polar cap i.e. polar gap",
    )
    parser.add_argument(
        "--extra_args",
        "-e",
        type=float,
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
        "--routine",
        "-r",
        type=str,
        nargs="?",
        default="north",
        const="north",
        choices=["north", "rotate", "translate"],
        help="plotting routine: defaults to north",
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


def valid_kernels(func_name: str) -> str:
    """
    check if valid kernel
    """
    if func_name in functions():
        return func_name
    else:
        raise ValueError("Not a valid kernel name to convolve")


def valid_plotting(func_name: str) -> str:
    """
    check if valid function
    """
    # check if valid function
    if func_name in functions():
        return func_name
    else:
        raise ValueError("Not a valid function name to plot")


def main():
    # initialise to None
    glm, glm_name = None, None

    args = read_args()
    flm_input = functions()[args.flm]
    glm_input = functions().get(args.convolve)
    # if not a convolution
    if glm_input is None:
        num_args = flm_input.__code__.co_argcount
        if args.extra_args is None or num_args == 0:
            flm, flm_name, config = flm_input()
        else:
            flm, flm_name, config = flm_input(args.extra_args)
    # if convolution then flm is a map so no extra args
    else:
        flm, flm_name, _ = flm_input()
        num_args = glm_input.__code__.co_argcount
        if args.extra_args is None or num_args == 0:
            glm, glm_name, config = glm_input()
        else:
            glm, glm_name, config = glm_input(args.extra_args)

    # if using input from argparse
    config["annotation"] = args.annotation
    config["routine"] = args.routine
    config["type"] = args.type

    sc = SiftingConvolution(flm, flm_name, config, glm, glm_name)
    sc.plot(args.alpha, args.beta, args.gamma)


if __name__ == "__main__":
    main()
