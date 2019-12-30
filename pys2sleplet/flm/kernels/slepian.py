from argparse import Namespace
from typing import List, Tuple

import numpy as np

from pys2sleplet.slepian.slepian_functions import SlepianFunctions

from ..functions import Functions


class Slepian(Functions):
    def __init__(
        self, L: int, args: Namespace = Namespace(extra_args=[0, 360, 0, 180, 0, 0, 0])
    ):
        self.rank, self.sf = self.validate_args(args)
        name = "slepian"
        self.reality = False
        super().__init__(name, L)

    @staticmethod
    def read_args(args: List[int]) -> Tuple[int, int]:
        # args
        try:
            rank = int(args.pop(0))
        except IndexError:
            # the most concentrated Slepian rank
            rank = 0
        try:
            order = int(args.pop(0))
        except IndexError:
            # D matrix corresponding to m=0 for polar cap
            order = 0
        return rank, order

    def validate_args(self, args) -> Tuple[int, SlepianFunctions]:
        extra_args = args.extra_args
        rank, order = self.read_args(extra_args)

        # validation
        if not float(rank).is_integer() or rank < 0:
            raise ValueError(f"Slepian concentration rank should be a positive integer")
        if sf.is_polar_cap:
            if rank >= self.L - abs(order):
                raise ValueError(
                    f"Slepian concentration rank should be less than {self.L - abs(order)}"
                )
        else:
            if rank >= self.L * self.L:
                raise ValueError(
                    f"Slepian concentration rank should be less than {self.L * self.L}"
                )
        rank = int(rank)
        return rank

    def _create_flm(self) -> np.ndarray:
        flm = self.sf.eigenvectors[self.rank]
        print(f"Eigenvalue {self.rank}: {self.sf.eigenvalues[self.rank]:e}")
        return flm
