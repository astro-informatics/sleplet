"""Contains the abstract `SlepianFunctions` class."""
import abc
import dataclasses
import logging

import numpy as np
import numpy.typing as npt
import pydantic

import sleplet._validation
from sleplet.slepian.region import Region

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.Validation)
class SlepianFunctions:
    """Abstract parent class of creating the different Slepian regions on the sphere."""

    L: int
    """The spherical harmonic bandlimit."""
    # TODO: adjust once https://github.com/pydantic/pydantic/issues/5470 fixed
    eigenvalues: npt.NDArray[np.float_] = dataclasses.field(
        default_factory=lambda: np.empty(0),
        kw_only=True,
        repr=True,
    )
    eigenvectors: npt.NDArray[np.complex_] = dataclasses.field(
        default_factory=lambda: np.empty(0, dtype=np.complex_),
        kw_only=True,
        repr=True,
    )
    mask: npt.NDArray[np.float_] = dataclasses.field(
        default_factory=lambda: np.empty(0),
        kw_only=True,
        repr=False,
    )
    matrix_location: str = dataclasses.field(default="", kw_only=True, repr=False)
    N: int = dataclasses.field(default=0, kw_only=True, repr=False)
    name: str = dataclasses.field(default="", kw_only=True, repr=False)
    region: Region = dataclasses.field(
        default_factory=lambda: Region(theta_max=0),
        kw_only=True,
        repr=False,
    )
    resolution: int = dataclasses.field(default=0, kw_only=True, repr=False)

    def __post_init_post_parse__(self) -> None:
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

    @abc.abstractmethod
    def _create_fn_name(self) -> str:
        """Creates the name for plotting."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_region(self) -> "sleplet.slepian.region.Region":
        """Creates a region object for area of interest."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_mask(self) -> npt.NDArray[np.float_]:
        """Creates a mask of the region of interest."""
        raise NotImplementedError

    @abc.abstractmethod
    def _calculate_area(self) -> float:
        """Calculates area of region."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_matrix_location(self) -> str:
        """Creates the name of the matrix binary."""
        raise NotImplementedError

    @abc.abstractmethod
    def _solve_eigenproblem(
        self,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        """Solves the eigenproblem for the given function."""
        raise NotImplementedError
