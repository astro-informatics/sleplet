"""Contains the abstract `MeshCoefficients` class."""
from abc import abstractmethod
from dataclasses import KW_ONLY

import numpy as np
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass

import sleplet._mask_methods
import sleplet._string_methods
import sleplet._validation
import sleplet._vars
from sleplet.meshes.mesh import Mesh

_COEFFICIENTS_TO_NOT_MASK: str = "slepian"


@dataclass(config=sleplet._validation.Validation)
class MeshCoefficients:
    """Abstract parent class to handle Fourier/Slepian coefficients on the mesh."""

    mesh: Mesh
    """A mesh object."""
    _: KW_ONLY
    extra_args: list[int] | None = None
    """Control the extra arguments for the given set of mesh
    coefficients. Only to be set by the `mesh` CLI."""
    noise: float | None = None
    """How much to noise the data."""
    region: bool = False
    """Whether to set a region or not, used in the Slepian case."""

    def __post_init_post_parse__(self) -> None:
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

    @validator("coefficients", check_fields=False)
    def _check_coefficients(cls, v, values):
        if (
            values["region"]
            and _COEFFICIENTS_TO_NOT_MASK not in cls.__class__.__name__.lower()
        ):
            v = sleplet._mask_methods.ensure_masked_bandlimit_mesh_signal(
                values["mesh"],
                v,
            )
        return v

    @abstractmethod
    def _add_noise_to_signal(
        self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """Adds Gaussian white noise to the signal."""
        raise NotImplementedError

    @abstractmethod
    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        """Creates the flm on the north pole."""
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> str:
        """Creates the name of the function."""
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        """
        Initialises function specific args
        either default value or user input.
        """
        raise NotImplementedError
