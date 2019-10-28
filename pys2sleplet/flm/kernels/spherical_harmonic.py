from typing import List

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.string_methods import filename_args


class SphericalHarmonic(Functions):
    def __init__(self, args: List[float] = [0.0, 0.0]):
        self.ell, self.m = self.validate_args(args)
        super().__init__(
            f"spherical_harmonic{filename_args(self.ell, 'l')}{filename_args(self.m, 'm')}"
        )

    @staticmethod
    def read_args(args):
        # args
        try:
            ell, m = args.pop(0), args.pop(0)
        except IndexError:
            raise ValueError("function requires exactly two extra args")
        return ell, m

    def validate_args(self, args):
        ell, m = self.read_args(args)

        # validation
        if ell < 0 or not ell.is_integer():
            raise ValueError("l should be a positive integer")
        if not m.is_integer() or abs(m) > ell:
            raise ValueError("m should be an integer |m| <= l")
        return ell, m

    def create_flm(self):
        flm = np.zeros((self.L * self.L), dtype=complex)
        ind = ssht.elm2ind(self.ell, self.m)
        flm[ind] = 1
        return flm
