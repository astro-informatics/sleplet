from typing import Dict, List, Optional

import numpy as np

from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.config import config
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.slepian_methods import choose_slepian_method


class Slepian(Functions):
    def __init__(self, L: int, args: List[int] = None) -> None:
        self.L = L
        self.reality = False
        self.phi_min = np.deg2rad(config.PHI_MIN)
        self.phi_max = np.deg2rad(config.PHI_MAX)
        self.theta_min = np.deg2rad(config.THETA_MIN)
        self.theta_max = np.deg2rad(config.THETA_MAX)
        self.s = choose_slepian_method()
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        if args is not None:
            num_args = 1
            if len(args) != num_args:
                raise ValueError(
                    f"The number of extra arguments should be 1 or {num_args}"
                )
            rank = args[0]
        else:
            rank = 0
        self.rank = rank

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

    @property
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
