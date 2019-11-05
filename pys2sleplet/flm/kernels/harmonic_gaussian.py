from typing import List

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.string_methods import filename_args


class HarmonicGaussian(Functions):
    def __init__(self, args: List[float] = [3.0, 3.0]):
        self.l_sig, self.m_sig = self.validate_args(args)
        super().__init__(
            f"harmonic_gaussian{filename_args(self.l_sig, 'lsig')}{filename_args(self.m_sig, 'msig')}"
        )

    @staticmethod
    def read_args(args):
        # args
        try:
            l_sig, m_sig = args.pop(0), args.pop(0)
        except IndexError:
            raise ValueError("function requires exactly two extra args")
        return l_sig, m_sig

    def validate_args(self, args):
        l_sig, m_sig = self.read_args(args)

        # validation
        if not l_sig.is_integer():
            raise ValueError("l sigma should be an integer")
        if not m_sig.is_integer():
            raise ValueError("m sigma should be an integer")
        l_sig, m_sig = 10 ** l_sig, 10 ** m_sig
        return l_sig, m_sig

    def create_flm(self):
        self.flm = np.zeros((self.L * self.L), dtype=complex)
        for ell in range(self.L):
            upsilon_l = np.exp(-((ell / self.l_sig) ** 2) / 2)
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                self.flm[ind] = upsilon_l * np.exp(-((m / self.m_sig) ** 2) / 2)
