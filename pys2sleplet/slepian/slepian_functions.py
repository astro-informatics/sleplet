from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pys2sleplet.utils.logger import logger


@dataclass  # type:ignore
class SlepianFunctions:
    L: int
    _area: float = field(init=False, repr=False)
    _eigenvalues: np.ndarray = field(init=False, repr=False)
    _eigenvectors: np.ndarray = field(init=False, repr=False)
    _L: int = field(init=False, repr=False)
    _mask: np.ndarray = field(init=False, repr=False)
    _matrix_location: Path = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _N: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._create_mask()
        self._create_fn_name()
        self._calculate_area()
        self.N = round(self.area * self.L ** 2 / (4 * np.pi))
        self._create_matrix_location()
        logger.info("start solving eigenproblem")
        self._solve_eigenproblem()
        logger.info("finished solving eigenproblem")

    @property
    def area(self) -> float:
        return self._area

    @area.setter
    def area(self, area: float) -> None:
        self._area = area

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

    @property  # type:ignore
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
    def N(self) -> int:
        return self._N

    @N.setter
    def N(self, N: int) -> None:
        self._N = N

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def _create_fn_name(self) -> None:
        """
        creates the name for plotting
        """
        raise NotImplementedError

    @abstractmethod
    def _create_mask(self) -> None:
        """
        creates a mask of the region of interest
        """
        raise NotImplementedError

    @abstractmethod
    def _calculate_area(self) -> None:
        """
        calculates area of region
        """
        raise NotImplementedError

    @abstractmethod
    def _create_matrix_location(self) -> None:
        """
        creates the name of the matrix binary
        """
        raise NotImplementedError

    @abstractmethod
    def _solve_eigenproblem(self) -> None:
        """
        solves the eigenproblem for the given function
        """
        raise NotImplementedError
