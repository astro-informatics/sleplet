from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class SphericalHarmonic(Functions):
    _ell: int = field(default=0, init=False, repr=False)
    _m: int = field(default=0, init=False, repr=False)

    def _create_annotations(self) -> List[Dict]:
        pass

    def _create_flm(self, L: int) -> np.ndarray:
        flm = np.zeros((L * L), dtype=complex)
        ind = ssht.elm2ind(self.ell, self.m)
        flm[ind] = 1
        return flm

    def _create_name(self) -> str:
        name = (
            "spherical_harmonic"
            f"{filename_args(self.ell, 'l')}"
            f"{filename_args(self.m, 'm')}"
        )
        return name

    def _set_reality(self) -> bool:
        return False

    def _setup_args(self, extra_args: Optional[List[int]]) -> None:
        if extra_args is not None:
            num_args = 2
            if len(extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.ell, self.m = extra_args

    @property
    def ell(self) -> int:
        return self._ell

    @ell.setter
    def ell(self, ell: int) -> None:
        if not isinstance(ell, int):
            raise TypeError("ell should be an integer")
        if ell < 0:
            raise ValueError("ell should be positive")
        if ell >= self.L:
            raise ValueError("ell should be less than or equal to L")
        self._ell = ell

    @property
    def m(self) -> int:
        return self._m

    @m.setter
    def m(self, m: int) -> None:
        if not isinstance(m, int):
            raise TypeError("m should be an integer")
        if abs(m) > self.ell:
            raise ValueError("the magnitude of m should be less than ell")
        self._m = m
