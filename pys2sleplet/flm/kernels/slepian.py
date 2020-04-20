from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import choose_slepian_method


@dataclass
class Slepian(Functions):
    L: int
    reality: bool = field(default=False)
    s: SlepianFunctions = field(init=False)
    __rank: int = field(default=0, init=False, repr=False)


def __post_init__(self) -> None:
    self.s = choose_slepian_method(self.L)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            num_args = 1
            if len(args) != num_args:
                raise ValueError(
                    f"The number of extra arguments should be 1 or {num_args}"
                )
            self.rank = args[0]

    def _create_name(self) -> str:
        name = f"{self.s.name}_rank{self.rank}"
        return name

    def _create_flm(self, L: int) -> np.ndarray:
        flm = self.s.eigenvectors[self.rank]
        logger.info(f"Eigenvalue {self.rank}: {self.s.eigenvalues[self.rank]:e}")
        return flm

    def _create_annotations(self) -> List[Dict]:
        annotations = self.s.annotations
        return annotations

    @property  # type: ignore
    def rank(self) -> int:
        return self.__rank

    @rank.setter
    def rank(self, var: int) -> None:
        if not isinstance(var, int):
            raise TypeError("rank should be an integer")
        if var < 0:
            raise ValueError("rank cannot be negative")
        if var >= self.L:
            raise ValueError(f"rank should be no more than {self.L}")
        self.__rank = var
