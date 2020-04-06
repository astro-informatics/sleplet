from dataclasses import InitVar, dataclass
from typing import Dict, List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class SphericalHarmonic(Functions):
    L: int
    args: Optional[List[int]] = None
    reality: InitVar[bool] = False

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            num_args = 2
            if len(args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
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

    def _create_annotations(self) -> List[Dict]:
        pass

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
