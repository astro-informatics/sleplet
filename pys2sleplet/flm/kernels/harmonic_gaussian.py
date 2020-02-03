from typing import List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.string_methods import filename_args

from ..functions import Functions


class HarmonicGaussian(Functions):
    def __init__(self, L: int, args: List[int] = None):
        self.reality = False
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            num_args = 2
            if len(args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            l_sigma, m_sigma = [10 ** x for x in args]
        else:
            l_sigma, m_sigma = 1_000, 1_000
        self.l_sigma, self.m_sigma = l_sigma, m_sigma

    def _create_flm(self, L: int) -> np.ndarray:
        flm = np.zeros((L * L), dtype=complex)
        for ell in range(L):
            upsilon_l = np.exp(-((ell / self.l_sigma) ** 2) / 2)
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                flm[ind] = upsilon_l * np.exp(-((m / self.m_sigma) ** 2) / 2)
        return flm

    def _create_name(self) -> str:
        name = f"harmonic_gaussian{filename_args(self.l_sigma, 'lsig')}{filename_args(self.m_sigma, 'msig')}"
        return name

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
