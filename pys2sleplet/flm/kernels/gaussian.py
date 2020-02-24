from typing import Dict, List, Optional

import numpy as np
import pyssht as ssht
from slepian.functions import Functions
from utils.string_methods import filename_args


class Gaussian(Functions):
    def __init__(self, L: int, args: List[int] = None) -> None:
        self.reality = True
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            num_args = 1
            if len(args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            sigma = 10 ** args[0]
        else:
            sigma = 1_000
        self.sigma = sigma

    def _create_flm(self, L: int) -> np.ndarray:
        flm = np.zeros((L * L), dtype=complex)
        for ell in range(L):
            ind = ssht.elm2ind(ell, m=0)
            flm[ind] = np.exp(-ell * (ell + 1) / (2 * self.sigma * self.sigma))
        return flm

    def _create_name(self) -> str:
        name = f"gaussian{filename_args(self.sigma, 'sig')}"
        return name

    def _create_annotations(self) -> List[Dict]:
        pass

    @property
    def sigma(self) -> float:
        return self.__sigma

    @sigma.setter
    def sigma(self, var: float) -> None:
        self.__sigma = var
