from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from pys2sleplet.utils.logger import logger


@dataclass  # type: ignore
class SlepianFunctions:
    L: int
    _annotations: List[Dict] = field(default_factory=list, init=False, repr=False)
    _eigenvalues: np.ndarray = field(init=False, repr=False)
    _eigenvectors: np.ndarray = field(init=False, repr=False)
    _L: int = field(init=False, repr=False)
    _matrix_location: Path = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._create_annotations()
        self.mask = self._create_mask(self.L)
        self.matrix_location = self._create_matrix_location(self.L)
        self.name = self._create_fn_name()
        logger.info("start solving eigenproblem")
        self.eigenvalues, self.eigenvectors = self._solve_eigenproblem(self.L)
        logger.info("finished solving eigenproblem")

    @property
    def annotations(self) -> List[Dict]:
        return self._annotations

    @annotations.setter
    def annotations(self, annotations: np.ndarray) -> None:
        self._annotations = annotations

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, eigenvalues: np.ndarray) -> None:
        self._eigenvalues = eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        return self._eigenvectors

    @eigenvectors.setter
    def eigenvectors(self, eigenvectors: np.ndarray) -> None:
        self._eigenvectors = eigenvectors

    @property  # type: ignore
    def L(self) -> int:
        return self._L

    @L.setter
    def L(self, L: int) -> None:
        self._L = L

    @property
    def mask(self) -> np.ndarray:
        return self._mask

    @mask.setter
    def mask(self, mask: np.ndarray) -> None:
        self._mask = mask

    @property
    def matrix_location(self) -> Path:
        return self._matrix_location

    @matrix_location.setter
    def matrix_location(self, matrix_location: Path) -> None:
        self._matrix_location = matrix_location

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def _create_annotations(self) -> None:
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
    def _create_mask(self, L: int) -> np.ndarray:
        """
        creates a mask of the region of interest
        """
        raise NotImplementedError

    @abstractmethod
    def _create_matrix_location(self, L: int) -> Path:
        """
        creates the name of the matrix binary
        """
        raise NotImplementedError

    @abstractmethod
    def _solve_eigenproblem(self, L: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        solves the eigenproblem for the given function
        """
        raise NotImplementedError
