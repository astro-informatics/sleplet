from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pys2sleplet.utils.vars import DC_VAR_NOT_INIT


@dataclass  # type: ignore
class SlepianFunctions:
    __annotations: List[Dict] = field(default_factory=list, init=False, repr=False)
    __eigenvalues: np.ndarray = DC_VAR_NOT_INIT
    __eigenvectors: np.ndarray = DC_VAR_NOT_INIT
    __L: int = DC_VAR_NOT_INIT
    __matrix_location: Path = DC_VAR_NOT_INIT
    __name: str = DC_VAR_NOT_INIT

    def __post_init__(self) -> None:
        self.annotations = self._create_annotations()
        self.name = self._create_fn_name()
        self.matrix_location = self._create_matrix_location()
        self.eigenvalues, self.eigenvectors = self._solve_eigenproblem()

    @property
    def annotations(self) -> List[Dict]:
        return self.__annotations

    @annotations.setter
    def annotations(self, annotations: np.ndarray) -> None:
        self.__annotations = annotations

    @property
    def eigenvalues(self) -> np.ndarray:
        return self.__eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, eigenvalues: np.ndarray) -> None:
        self.__eigenvalues = eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        return self.__eigenvectors

    @eigenvectors.setter
    def eigenvectors(self, eigenvectors: np.ndarray) -> None:
        self.__eigenvectors = eigenvectors

    @property
    def L(self) -> int:
        return self.__L

    @L.setter
    def L(self, L: int) -> None:
        self.__L = L

    @property
    def matrix_location(self) -> Path:
        return self.__matrix_location

    @matrix_location.setter
    def matrix_location(self, matrix_location: Path) -> None:
        self.__matrix_location = matrix_location

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name: str) -> None:
        self.__name = name

    @abstractmethod
    def _create_annotations(self) -> List[Dict]:
        """
        creates the annotations for the plot
        """
        raise NotImplementedError

    @abstractmethod
    def _create_fn_name(self) -> str:
        """
        creates the name for plotting
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
