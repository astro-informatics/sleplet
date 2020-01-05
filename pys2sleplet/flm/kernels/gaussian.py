from typing import List

import numpy as np
import pyssht as ssht

from pys2sleplet.utils.string_methods import filename_args, verify_args

from ..functions import Functions


class Gaussian(Functions):
    def __init__(self, L: int, args: List[int] = None):
        self.reality = True
        if args is not None:
            verify_args(args, 1)
            self.__sigma = 10 ** args[0]
        else:
            self.__sigma = 1e3
        super().__init__(L, args)

    def _create_flm(self, L: int) -> np.ndarray:
        flm = np.zeros((L * L), dtype=complex)
        for ell in range(L):
            ind = ssht.elm2ind(ell, m=0)
            flm[ind] = np.exp(-ell * (ell + 1) / (2 * self.sigma * self.sigma))
        return flm

    def _create_name(self) -> str:
        name = f"gaussian{filename_args(self.sigma, 'sig')}"
        return name

    @property
    def sigma(self) -> float:
        return self.__sigma

    @sigma.setter
    def sigma(self, var: int) -> None:
        self.__sigma = 10 ** var
