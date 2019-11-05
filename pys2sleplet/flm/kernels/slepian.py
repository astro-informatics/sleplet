from typing import List

from pys2sleplet.flm.functions import Functions
from pys2sleplet.slepian import SlepianFunctions


class Slepian(Functions):
    def __init__(self, args: List[float] = [0.0, 360.0, 0.0, 40.0, 0.0, 0.0, 0.0]):
        self.rank, self.sf = self.validate_args(args)
        super().__init__(
            f"slepian{self.sf.filename_angle()}{self.sf.filename}_rank{self.rank}",
            reality=False,
        )

    @staticmethod
    def read_args(args):
        # args
        try:
            phi_min, phi_max, theta_min, theta_max = (
                args.pop(0),
                args.pop(0),
                args.pop(0),
                args.pop(0),
            )
        except IndexError:
            raise ValueError("function requires at least four extra args")
        try:
            rank = args.pop(0)
        except IndexError:
            rank = 0.0  # the most concentrated Slepian rank
        try:
            order = args.pop(0)
        except IndexError:
            order = 0.0  # D matrix corresponding to m=0 for polar cap
        try:
            double = args.pop(0)
        except IndexError:
            double = 0.0  # set boolean switch for polar gap off
        return phi_min, phi_max, theta_min, theta_max, rank, order, double

    def validate_args(self, args):
        phi_min, phi_max, theta_min, theta_max, rank, order, double = self.read_args(
            args
        )

        # initialise class
        sf = SlepianFunctions(
            self.L,
            phi_min,
            phi_max,
            theta_min,
            theta_max,
            order,
            double,
            ncpu,
            save_matrices,
        )

        # validation
        if not rank.is_integer() or rank < 0:
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
