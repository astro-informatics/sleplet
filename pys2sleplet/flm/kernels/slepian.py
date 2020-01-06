from typing import List, Optional

import numpy as np

from ..functions import Functions


class Slepian(Functions):
    # [0, 360, 0, 180, 0, 0, 0]
    def __init__(self, L: int, args: List[int] = None):
        self.__rank = 0
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        raise NotImplementedError

    def _create_name(self) -> str:
        name = "slepian"
        return name

    def _create_flm(self, L: int) -> np.ndarray:
        flm = self.eigenvectors[self.rank]
        print(f"Eigenvalue {self.rank}: {self.eigenvalues[self.rank]:e}")
        return flm

    @property
    def rank(self) -> int:
        return self.__rank

    @rank.setter
    def rank(self, var: int) -> None:
        self.__rank = var

    @property
    def eigenvectors(self) -> np.ndarray:
        return self.__eigenvectors

    @eigenvectors.setter
    def eigenvectors(self, var: np.ndarray) -> None:
        self.__eigenvectors = var

    @property
    def eigenvalues(self) -> np.ndarray:
        return self.__eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, var: np.ndarray) -> None:
        self.__eigenvalues = var
