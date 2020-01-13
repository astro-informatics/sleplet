from abc import abstractmethod
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


class SlepianFunctions:
    def __init__(self, L: int) -> None:
        self.__matrix_name = f"D_L-{L}_"
        self.__name = "slepian"
        self.L = L
        self.annotations = self._create_annotations()
        self.matrix_location = self._create_matrix_location()
        self.eigenvalues, self.eigenvectors = self._solve_eigenproblem()

    @abstractmethod
    def _create_annotations(self) -> List[Dict]:
        """
        creates the annotations for the plot
        """
        raise NotImplementedError

    @abstractmethod
    def _create_matrix_location(self) -> Path:
        """
        creates the name of the matrix binary
        """
        raise NotImplementedError

    @abstractmethod
    def _solve_eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        solves the eigenproblem for the given function
        """
        raise NotImplementedError

    @property
    def L(self) -> int:
        return self.__L

    @L.setter
    def L(self, var: int) -> None:
        self.__L = var

    @property
    def matrix_location(self) -> Path:
        return self.__matrix_location

    @matrix_location.setter
    def matrix_location(self, var: Path) -> None:
        self.__matrix_location = var

    @property
    def annotations(self) -> List[Dict]:
        return self.__annotations

    @annotations.setter
    def annotations(self, var: np.ndarray) -> None:
        self.__annotations = var

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
