from typing import List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.string_methods import filename_args, verify_args

from ..functions import Functions


class SphericalHarmonic(Functions):
    def __init__(self, L: int, args: List[int] = None):
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            verify_args(args, 2)
            ell, m = args
        else:
            ell, m = 0, 0
        self.ell, self.m = ell, m

    def _create_flm(self, L: int) -> np.ndarray:
        flm = np.zeros((L * L), dtype=complex)
        ind = ssht.elm2ind(self.ell, self.m)
        flm[ind] = 1
        return flm

    def _create_name(self) -> str:
        name = f"spherical_harmonic{filename_args(self.ell, 'l')}{filename_args(self.m, 'm')}"
        return name

    @property
    def ell(self) -> int:
        return self.__ell

    @ell.setter
    def ell(self, var: int) -> None:
        if not isinstance(var, int):
            raise TypeError("ell should be an integer")
        if var < 0:
            raise ValueError("ell should be positive")
        self.__ell = var

    @property
    def m(self) -> int:
        return self.__m

    @m.setter
    def m(self, var: int) -> None:
        if not isinstance(var, int):
            raise TypeError("m should be an integer")
        if abs(var) > self.ell:
            raise ValueError("the magnitude of m should be less than ell")
        self.__m = var
