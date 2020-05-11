from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class Gaussian(Functions):
    L: int
    extra_args: List[int]
    __sigma: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.reality = True
        self.sigma = 1_000

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            num_args = 1
            if len(args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.sigma = 10 ** args[0]

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
    def sigma(self, sigma: float) -> None:
        self.__sigma = sigma
