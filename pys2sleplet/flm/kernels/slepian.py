from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.slepian.slepian_functions import SlepianFunctions
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import choose_slepian_method


@dataclass
class Slepian(Functions):
    _rank: int = field(default=0, init=False, repr=False)
    _slepian: SlepianFunctions = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.slepian = choose_slepian_method()
        super().__post_init__()

    def _create_annotations(self) -> List[Dict]:
        annotations = self.slepian.annotations
        return annotations

    def _create_name(self) -> str:
        name = f"{self.slepian.name}_rank{self.rank}"
        return name

    def _create_flm(self, L: int) -> np.ndarray:
        flm = self.slepian.eigenvectors[self.rank]
        logger.info(f"Eigenvalue {self.rank}: {self.slepian.eigenvalues[self.rank]:e}")
        return flm

    def _set_reality(self) -> bool:
        return False

    def _setup_args(self, extra_args: Optional[List[int]]) -> None:
        if extra_args is not None:
            num_args = 1
            if len(extra_args) != num_args:
                raise ValueError(
                    f"The number of extra arguments should be 1 or {num_args}"
                )
            self.rank = extra_args[0]

    @property
    def rank(self) -> int:
        return self._rank

    @rank.setter
    def rank(self, rank: int) -> None:
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        if rank >= self.L:
            raise ValueError(f"rank should be no more than {self.L}")
        self._rank = rank

    @property
    def slepian(self) -> SlepianFunctions:
        return self._slepian

    @slepian.setter
    def slepian(self, slepian: SlepianFunctions) -> None:
        self._slepian = slepian
