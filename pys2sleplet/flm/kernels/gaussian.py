from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class Gaussian(Functions):
    _sigma: float = field(default=1_000, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self, L: int) -> np.ndarray:
        flm = np.zeros((L * L), dtype=complex)
        for ell in range(L):
            ind = ssht.elm2ind(ell, m=0)
            flm[ind] = np.exp(-ell * (ell + 1) / (2 * self.sigma * self.sigma))
        return flm

    def _create_name(self) -> str:
        name = f"gaussian{filename_args(self.sigma, 'sig')}"
        return name

    def _set_reality(self) -> bool:
        return True

    def _setup_args(self, extra_args: Optional[List[int]]) -> None:
        if extra_args is not None:
            num_args = 1
            if len(extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.sigma = 10 ** extra_args[0]

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, sigma: float) -> None:
        self._sigma = sigma
