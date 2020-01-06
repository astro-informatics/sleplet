from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from ..functions import Functions


class Slepian(Functions):
    # [0, 360, 0, 180, 0, 0, 0]
    def __init__(self, L: int, args: List[int] = None):
        self.__rank = 0
        self.__matrix_name = f"D_L-{L}_"
        self.__eigenvalues, self.__eigenvectors = self.eigenproblem()
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

    @property
    def matrix_name(self) -> str:
        return self.__matrix_name

    @matrix_name.setter
    def matrix_name(self, var: str) -> None:
        self.__matrix_name = var

    @abstractmethod
    def eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def annotations(self) -> List[dict]:
        raise NotImplementedError
