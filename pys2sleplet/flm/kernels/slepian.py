from argparse import Namespace
from typing import List, Tuple

from ...slepian.slepian_functions import SlepianFunctions
from ..functions import Functions


class Slepian(Functions):
    def __init__(
        self, L: int, args: Namespace = Namespace(extra_args=[0, 360, 0, 40, 0, 0, 0])
    ):
        self.rank, self.sf = self.validate_args(args)
        name = f"slepian{self.sf.filename_angle()}{self.sf.filename}_rank{self.rank}"
        super().__init__(name, L)

    @staticmethod
    def read_args(args: List[int]) -> Tuple[int, int, int, int, int, int, int]:
        # args
        try:
            phi_min, phi_max, theta_min, theta_max = (
                int(args.pop(0)),
                int(args.pop(0)),
                int(args.pop(0)),
                int(args.pop(0)),
            )
        except IndexError:
            raise ValueError("function requires at least four extra args")
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
        try:
            double = int(args.pop(0))
        except IndexError:
            # set boolean switch for polar gap off
            double = 0
        return phi_min, phi_max, theta_min, theta_max, rank, order, double

    def validate_args(self, args) -> Tuple[int, SlepianFunctions]:
        extra_args = args.extra_args
        phi_min, phi_max, theta_min, theta_max, rank, order, double = self.read_args(
            extra_args
        )

        # initialise class
        sf = SlepianFunctions(
            self.L, phi_min, phi_max, theta_min, theta_max, order, double
        )

        # validation
        if not float(rank).is_integer() or rank < 0:
            raise ValueError(f"Slepian concentration rank should be a positive integer")
        if sf.is_polar_cap:
            if rank >= self.L - abs(sf.order):
                raise ValueError(
                    f"Slepian concentration rank should be less than {self.L - abs(sf.order)}"
                )
        else:
            if rank >= self.L * self.L:
                raise ValueError(
                    f"Slepian concentration rank should be less than {self.L * self.L}"
                )
        rank = int(rank)
        return rank, sf

    def create_flm(self):
        self.flm = self.sf.eigenvectors[self.rank]
        print(f"Eigenvalue {self.rank}: {self.sf.eigenvalues[self.rank]:e}")
