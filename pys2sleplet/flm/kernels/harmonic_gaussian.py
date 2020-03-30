from typing import Dict, List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.string_methods import filename_args


class HarmonicGaussian(Functions):
    def __init__(self, L: int, args: Optional[List[int]] = None) -> None:
        self.reality = False
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            num_args = 2
            if len(args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.l_sigma, self.m_sigma = [10 ** x for x in args]
        else:
            self.l_sigma, self.m_sigma = 1e3, 1e3

    def _create_flm(self, L: int) -> np.ndarray:
        flm = np.zeros((L * L), dtype=complex)
        for ell in range(L):
            upsilon_l = np.exp(-((ell / self.l_sigma) ** 2) / 2)
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                flm[ind] = upsilon_l * np.exp(-((m / self.m_sigma) ** 2) / 2)
        return flm

    def _create_name(self) -> str:
        name = (
            "harmonic_gaussian"
            f"{filename_args(self.l_sigma, 'lsig')}"
            f"{filename_args(self.m_sigma, 'msig')}"
        )
        return name

    def _create_annotations(self) -> List[Dict]:
        pass

    @property
    def l_sigma(self) -> float:
        return self.__l_sigma

    @l_sigma.setter
    def l_sigma(self, var: float) -> None:
        self.__l_sigma = var

    @property
    def m_sigma(self) -> float:
        return self.__m_sigma

    @m_sigma.setter
    def m_sigma(self, var: float) -> None:
        self.__m_sigma = var
