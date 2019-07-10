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
        "--rank",
        "-r",
        type=valid_range,
        default=1,
        help="retrieve the Slepian coefficients descending from 1 down to rank",
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
    parser.add_argument(
        "--annotation",
        "-n",
        action="store_false",
        help="flag which if passed removes any annotation",
    )

    args = parser.parse_args()
    return args


def valid_range(rank: int) -> int:
    config = asdict(Config())
    L, rank = config["L"], int(rank)
    # check if valid range
    if rank <= L * L and rank > 0:
        return rank
    else:
        raise ValueError(f"Must be positive and less than {L * L} coefficients")


if __name__ == "__main__":
    config = asdict(Config())
    args = read_args()

    # if using input from argparse
    config["annotation"] = args.annotation
    config["type"] = args.type

    sf = SlepianFunctions(
        args.phi_min, args.phi_max, args.theta_min, args.theta_max, config
    )
    for i in range(args.rank):
        sf.plot(i)
