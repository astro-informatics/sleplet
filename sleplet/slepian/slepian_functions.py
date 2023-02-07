from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sleplet.utils.logger import logger


@dataclass
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
        self.N = round(self.area * self.L**2 / (4 * np.pi))
        logger.info(f"Shannon number N={self.N}")
        self._create_matrix_location()
        logger.info("start solving eigenproblem")
        self._solve_eigenproblem()
        logger.info("finished solving eigenproblem")

    @area.setter
    def area(self, area: float) -> None:
        self._area = area

    @eigenvalues.setter
    def eigenvalues(self, eigenvalues: np.ndarray) -> None:
        self._eigenvalues = eigenvalues

    @eigenvectors.setter
    def eigenvectors(self, eigenvectors: np.ndarray) -> None:
        self._eigenvectors = eigenvectors

    @L.setter
    def L(self, L: int) -> None:
        self._L = L

    @mask.setter
    def mask(self, mask: np.ndarray) -> None:
        self._mask = mask

    @matrix_location.setter
    def matrix_location(self, matrix_location: Path) -> None:
        self._matrix_location = matrix_location

    @N.setter
    def N(self, N: int) -> None:
        self._N = N

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
