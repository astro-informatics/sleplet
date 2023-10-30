"""Contains the abstract `SlepianFunctions` class."""
import abc
import dataclasses
import logging

import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._validation
from sleplet.slepian.region import Region

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class SlepianFunctions:
    """Abstract parent class of creating the different Slepian regions on the sphere."""

    L: int
    """The spherical harmonic bandlimit."""
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
    _resolution: int = dataclasses.field(
        default=0,
        kw_only=True,
        repr=False,
    )

    def __post_init__(self: typing_extensions.Self) -> None:
        self.region = self._create_region()
        self.mask = self._create_mask()
        self.name = self._create_fn_name()
        area = self._calculate_area()
        self.N = round(area * self.L**2 / (4 * np.pi))
        msg = f"Shannon number N={self.N}"
        _logger.info(msg)
        self.matrix_location = self._create_matrix_location()
        _logger.info("start solving eigenproblem")
        self.eigenvalues, self.eigenvectors = self._solve_eigenproblem()
        _logger.info("finished solving eigenproblem")

    @abc.abstractmethod
    def _create_fn_name(self: typing_extensions.Self) -> str:
        """Create the name for plotting."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_region(self: typing_extensions.Self) -> "sleplet.slepian.region.Region":
        """Create a region object for area of interest."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_mask(self: typing_extensions.Self) -> npt.NDArray[np.float_]:
        """Create a mask of the region of interest."""
        raise NotImplementedError

    @abc.abstractmethod
    def _calculate_area(self: typing_extensions.Self) -> float:
        """Calculate area of region."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_matrix_location(self: typing_extensions.Self) -> str:
        """Create the name of the matrix binary."""
        raise NotImplementedError

    @abc.abstractmethod
    def _solve_eigenproblem(
        self: typing_extensions.Self,
    ) -> tuple[npt.NDArray[np.float_], npt.NDArray[np.complex_]]:
        """Solve the eigenproblem for the given function."""
        raise NotImplementedError
