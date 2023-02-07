from abc import abstractmethod
from dataclasses import KW_ONLY, field

import numpy as np
from pydantic.dataclasses import dataclass

from sleplet.utils.logger import logger
from sleplet.utils.region import Region
from sleplet.utils.validation import Validation


@dataclass(config=Validation)
class SlepianFunctions:
    L: int
    _: KW_ONLY
    area: np.ndarray = field(init=False, repr=False)
    eigenvalues: np.ndarray = field(init=False, repr=False)
    eigenvectors: np.ndarray = field(init=False, repr=False)
    mask: np.ndarray = field(init=False, repr=False)
    name: str = field(init=False, repr=False)
    region: Region = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._create_mask()
        self._create_fn_name()
        self._calculate_area()
        self.N = np.round(self.area * self.L**2 / (4 * np.pi))
        logger.info(f"Shannon number N={self.N}")
        self._create_matrix_location()
        logger.info("start solving eigenproblem")
        self._solve_eigenproblem()
        logger.info("finished solving eigenproblem")

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
