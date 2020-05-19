from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class SphericalHarmonic(Functions):
    L: int
    extra_args: Optional[List[int]] = field(default=None)
    _ell: int = field(default=0, init=False, repr=False)
    _m: int = field(default=0, init=False, repr=False)

    def _setup_args(self) -> None:
        if self.extra_args is not None:
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.ell, self.m = self.extra_args

    def _set_reality(self) -> bool:
        return False

    def _create_flm(self) -> np.ndarray:
        flm = np.zeros((self.L * self.L), dtype=complex)
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
        return self._ell

    @ell.setter
    def ell(self, ell: int) -> None:
        if not isinstance(ell, int):
            raise TypeError("ell should be an integer")
        if ell < 0:
            raise ValueError("ell should be positive")
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
