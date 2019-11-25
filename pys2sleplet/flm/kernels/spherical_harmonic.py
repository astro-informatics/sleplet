from argparse import Namespace
from typing import List, Tuple

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.string_methods import filename_args


class SphericalHarmonic(Functions):
    def __init__(self, L: int, args: Namespace = Namespace(extra_args=[0, 0])):
        self.ell, self.m = self.validate_args(args)
        name = f"spherical_harmonic{filename_args(self.ell, 'l')}{filename_args(self.m, 'm')}"
        reality = False
        super().__init__(name, L, reality)

    @staticmethod
    def read_args(args: List[int]) -> Tuple[int, int]:
        # args
        try:
            ell, m = int(args.pop(0)), int(args.pop(0))
        except IndexError:
            raise ValueError("function requires exactly two extra args")
        return ell, m

    def validate_args(self, args: Namespace) -> Tuple[int, int]:
        extra_args = args.extra_args
        ell, m = self.read_args(extra_args)

        # validation
        if ell < 0 or not float(ell).is_integer():
            raise ValueError("l should be a positive integer")
        if not float(m).is_integer() or abs(m) > ell:
            raise ValueError("m should be an integer |m| <= l")
        return ell, m

    def create_flm(self) -> np.ndarray:
        flm = np.zeros((self.L * self.L), dtype=complex)
        ind = ssht.elm2ind(self.ell, self.m)
        flm[ind] = 1
        return flm
