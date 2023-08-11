"""Contains the abstract `MeshCoefficients` class."""
import abc
import dataclasses

import numpy as np
import numpy.typing as npt
import pydantic

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
    # TODO: adjust once https://github.com/pydantic/pydantic/issues/5470 fixed
    coefficients: npt.NDArray[np.complex_ | np.float_] = dataclasses.field(
        default_factory=lambda: np.empty((697, 2790)),
        repr=False,
    )
    name: str = dataclasses.field(default="", repr=False)
    snr: float | None = dataclasses.field(default=None, repr=False)
    unnoised_coefficients: npt.NDArray[
        np.complex_ | np.float_
    ] | None = dataclasses.field(
        default=None,
        repr=False,
    )
    wavelets: npt.NDArray[np.float_] = dataclasses.field(
        default_factory=lambda: np.empty(0),
        repr=False,
    )

    def __post_init__(self) -> None:
        self._setup_args()
        self.name = self._create_name()
        self.coefficients = self._create_coefficients()
        self._add_details_to_name()
        self.unnoised_coefficients, self.snr = self._add_noise_to_signal()

    def _add_details_to_name(self) -> None:
        """Adds region to the name if present if not a Slepian function."""
        if self.region and "slepian" not in self.mesh.name:
            self.name += "_region"
        if self.noise is not None:
            self.name += f"{sleplet._string_methods.filename_args(self.noise, 'noise')}"
        if self.mesh.zoom:
            self.name += "_zoom"

    @pydantic.field_validator("coefficients", check_fields=False)
    def _check_coefficients(cls, v, info: pydantic.FieldValidationInfo):
        if (
            info.data["region"]
            and _COEFFICIENTS_TO_NOT_MASK not in cls.__class__.__name__.lower()
        ):
            v = sleplet._mask_methods.ensure_masked_bandlimit_mesh_signal(
                info.data["mesh"],
                v,
            )
        return v

    @abc.abstractmethod
    def _add_noise_to_signal(
        self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """Adds Gaussian white noise to the signal."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        """Creates the flm on the north pole."""
        raise NotImplementedError

    @abc.abstractmethod
    def _create_name(self) -> str:
        """Creates the name of the function."""
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_args(self) -> None:
        """
        Initialises function specific args
        either default value or user input.
        """
        raise NotImplementedError
