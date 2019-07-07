#!/usr/bin/env python
from config import Config
from slepian_functions import SlepianFunctions
from argparse import ArgumentParser, Namespace
from dataclasses import asdict
import os


global __location__
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def read_args() -> Namespace:
    parser = ArgumentParser(description="Create SSHT plot")
    parser.add_argument(
        "rank",
        type=int,
        nargs="+",
        help="Slepian concetration rank - descending from 1",
    )
    parser.add_argument(
        "--phi_min",
        "-pmin",
        type=int,
        default=0,
        help="phi minimum in degrees for region - defaults to 0",
    )
    parser.add_argument(
        "--phi_max",
        "-pmax",
        type=int,
        default=360,
        help="phi maximum in degrees for region - defaults to 360",
    )
    parser.add_argument(
        "--theta_min",
        "-tmin",
        type=int,
        default=0,
        help="theta minimum in degrees for region - defaults to 0",
    )
    parser.add_argument(
        "--theta_max",
        "-tmax",
        type=int,
        default=40,
        help="theta maximum in degrees for region - defaults to 40",
    )
    parser.add_argument(
        "--type",
        "-t",
        type=str,
        nargs="?",
        default="abs",
        const="abs",
        choices=["abs", "real", "imag", "sum"],
        help="plotting type: defaults to abs",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = asdict(Config())
    args = read_args()

    # if using input from argparse
    config["type"] = args.type

    sf = SlepianFunctions(
        args.phi_min, args.phi_max, args.theta_min, args.theta_max, config
    )
    for i in args.rank:
        sf.plot(i)
