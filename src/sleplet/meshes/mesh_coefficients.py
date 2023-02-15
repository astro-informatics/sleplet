from abc import abstractmethod
from dataclasses import KW_ONLY

import numpy as np
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass
from sleplet.meshes.classes.mesh import Mesh
from sleplet.utils.mask_methods import ensure_masked_bandlimit_mesh_signal
from sleplet.utils.string_methods import filename_args
from sleplet.utils.validation import Validation

COEFFICIENTS_TO_NOT_MASK: str = "slepian"


@dataclass(config=Validation)
class MeshCoefficients:
    mesh: Mesh
    _: KW_ONLY
    extra_args: list[int] | None = None
    noise: float | None = None
    region: bool = False

    def __post_init_post_parse__(self) -> None:
        self._setup_args()
        self.name = self._create_name()
        self.coefficients = self._create_coefficients()
        self._add_details_to_name()
        self.unnoised_coefficients, self.snr = self._add_noise_to_signal()

    def _add_details_to_name(self) -> None:
        """
        adds region to the name if present if not a Slepian function
        """
        if self.region and "slepian" not in self.mesh.name:
            self.name += "_region"
        if self.noise is not None:
            self.name += f"{filename_args(self.noise, 'noise')}"
        if self.mesh.zoom:
            self.name += "_zoom"

    @validator("coefficients", check_fields=False)
    def check_coefficients(cls, v, values):
        if (
            values["region"]
            and COEFFICIENTS_TO_NOT_MASK not in cls.__class__.__name__.lower()
        ):
            v = ensure_masked_bandlimit_mesh_signal(values["mesh"], v)
        return v

    @abstractmethod
    def _add_noise_to_signal(
        self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """
        adds Gaussian white noise to the signal
        """
        raise NotImplementedError

    @abstractmethod
    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        """
        creates the flm on the north pole
        """
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> str:
        """
        creates the name of the function
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        """
        initialises function specific args
        either default value or user input
        """
        raise NotImplementedError
