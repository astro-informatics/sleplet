from typing import List

import numpy as np
import pyssht as ssht

from ...utils.string_methods import filename_args
from ..functions import Functions


class Gaussian(Functions):
    def __init__(self, args: List[float] = [3]):
        self.sig = self.validate_args(args)
        super().__init__(f"gaussian{filename_args(self.sig, 'sig')}", reality=True)

    @staticmethod
    def validate_args(args):
        # validation
        if not args[0].is_integer():
            raise ValueError("sigma should be an integer")
        sig = 10 ** args[0]
        return sig

    def create_flm(self):
        self.flm = np.zeros((self.L * self.L), dtype=complex)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, m=0)
            self.flm[ind] = np.exp(-ell * (ell + 1) / (2 * self.sig * self.sig))
