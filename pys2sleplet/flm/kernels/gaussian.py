from argparse import Namespace

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.string_methods import filename_args

from ..functions import Functions


class Gaussian(Functions):
    def __init__(self, L: int, args: Namespace = Namespace(extra_args=[3])):
        self.sig = self.validate_args(args)
        name = f"gaussian{filename_args(self.sig, 'sig')}"
        self.reality = True
        super().__init__(name, L)

    @staticmethod
    def validate_args(args: Namespace) -> int:
        extra_args = args.extra_args[0]
        # validation
        if not float(extra_args).is_integer():
            raise ValueError("sigma should be an integer")
        sig = 10 ** extra_args
        return sig

    def _create_flm(self) -> np.ndarray:
        flm = np.zeros((self.L * self.L), dtype=complex)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, m=0)
            flm[ind] = np.exp(-ell * (ell + 1) / (2 * self.sig * self.sig))
        return flm
