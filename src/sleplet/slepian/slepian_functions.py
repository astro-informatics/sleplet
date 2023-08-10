"""Contains the abstract `SlepianFunctions` class."""
import dataclasses
import logging
from abc import abstractmethod

import numpy as np
import pydantic
from numpy import typing as npt

import sleplet._validation

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class SlepianFunctions:
    """Abstract parent class of creating the different Slepian regions on the sphere."""

    L: int
    """The spherical harmonic bandlimit."""
    # TODO: adjust once https://github.com/pydantic/pydantic/issues/5470 fixed
    _: dataclasses.KW_ONLY
    resolution: int = dataclasses.field(default=1, repr=False)

    def __post_init__(self) -> None:
        self.region = self._create_region()
        self.mask = self._create_mask()
        self.name = self._create_fn_name()
        area = self._calculate_area()
        self.N = round(area * self.L**2 / (4 * np.pi))
        _logger.info(f"Shannon number N={self.N}")
        self.matrix_location = self._create_matrix_location()
        _logger.info("start solving eigenproblem")
        self.eigenvalues, self.eigenvectors = self._solve_eigenproblem()
        _logger.info("finished solving eigenproblem")

    @abstractmethod
    def _create_fn_name(self) -> str:
        """Creates the name for plotting."""
        raise NotImplementedError

    @abstractmethod
    def _create_region(self) -> "sleplet.slepian.region.Region":
        """Creates a region object for area of interest."""
        raise NotImplementedError

    @abstractmethod
    def _create_mask(self) -> npt.NDArray[np.float_]:
        """Creates a mask of the region of interest."""
        raise NotImplementedError

    @abstractmethod
    def _calculate_area(self) -> float:
        """Calculates area of region."""
        raise NotImplementedError

    @abstractmethod
    def _create_matrix_location(self) -> str:
        """Creates the name of the matrix binary."""
        raise NotImplementedError

    @abstractmethod
    def _solve_eigenproblem(
        self,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        """Solves the eigenproblem for the given function."""
        raise NotImplementedError
