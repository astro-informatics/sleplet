"""Contains the abstract `MeshCoefficients` class."""
import abc
import dataclasses

import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._mask_methods
import sleplet._string_methods
import sleplet._validation
import sleplet._vars
from sleplet.meshes.mesh import Mesh

_COEFFICIENTS_TO_NOT_MASK: str = "slepian"


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class MeshCoefficients:
    """Abstract parent class to handle Fourier/Slepian coefficients on the mesh."""

    mesh: Mesh
    """A mesh object."""
    _: dataclasses.KW_ONLY
    extra_args: list[int] | None = None
    """Control the extra arguments for the given set of mesh
    coefficients. Only to be set by the `mesh` CLI."""
    noise: float | None = None
    """How much to noise the data."""
    region: bool = False
    """Whether to set a region or not, used in the Slepian case."""
    _unnoised_coefficients: (
        npt.NDArray[np.complex_ | np.float_] | None
    ) = pydantic.Field(
        default=None,
        init_var=False,
        repr=False,
    )
    coefficients: npt.NDArray[np.complex_ | np.float_] = pydantic.Field(
        default_factory=lambda: np.empty(0),
        init_var=False,
        repr=False,
    )
    name: str = pydantic.Field(default="", init_var=False, repr=False)
    snr: float | None = pydantic.Field(default=None, init_var=False, repr=False)
    wavelet_coefficients: npt.NDArray[np.complex_ | np.float_] = pydantic.Field(
        default_factory=lambda: np.empty(0),
        init_var=False,
        repr=False,
    )
    wavelets: npt.NDArray[np.float_] = pydantic.Field(
        default_factory=lambda: np.empty(0),
        init_var=False,
        repr=False,
    )

    def __post_init__(self: typing_extensions.Self) -> None:
        self._setup_args()
        self.name = self._create_name()
        self.coefficients = self._create_coefficients()
        self._add_details_to_name()
        self._unnoised_coefficients, self.snr = self._add_noise_to_signal()

    def _add_details_to_name(self: typing_extensions.Self) -> None:
        """Add region to the name if present if not a Slepian function."""
        if self.region and "slepian" not in self.mesh.name:
            self.name += "_region"
        if self.noise is not None:
            self.name += f"{sleplet._string_methods.filename_args(self.noise, 'noise')}"
        if self.mesh.zoom:
            self.name += "_zoom"

    @abc.abstractmethod
    def _add_noise_to_signal(
        self: typing_extensions.Self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """Add Gaussian white noise to the signal."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        """Create the flm on the north pole."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_name(self: typing_extensions.Self) -> str:
        """Create the name of the function."""
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_args(self: typing_extensions.Self) -> None:
        """Initialise function specific args either default value or user input."""
        raise NotImplementedError
