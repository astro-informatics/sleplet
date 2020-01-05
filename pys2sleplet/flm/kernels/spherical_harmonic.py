from typing import List

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.string_methods import filename_args, verify_args

from ..functions import Functions


class SphericalHarmonic(Functions):
    def __init__(self, L: int, args: List[int] = None):
        if args is not None:
            verify_args(args, 2)
            self.__ell, self.__m = args
        else:
            self.__ell, self.__m = 0, 0
        super().__init__(L)

    def _create_flm(self, L: int) -> np.ndarray:
        flm = np.zeros((L * L), dtype=complex)
        ind = ssht.elm2ind(self.ell, self.m)
        flm[ind] = 1
        return flm

    def _create_name(self) -> str:
        name = f"spherical_harmonic{filename_args(self.ell, 'l')}{filename_args(self.m, 'm')}"
        return name

    @property
    def ell(self) -> float:
        return self.__ell

    @ell.setter
    def ell(self, var: int) -> None:
        if var < 0:
            raise ValueError("ell should be positive")
        self.__ell = var

    @property
    def m(self) -> float:
        return self.__m

    @m.setter
    def m(self, var: int) -> None:
        if abs(var) > self.ell:
            raise ValueError("the magnitude of m should be less than ell")
        self.__ell = var
