from abc import abstractmethod
from typing import Optional

import numpy as np
from pydantic.dataclasses import dataclass

from sleplet.meshes.classes.mesh import Mesh
from sleplet.utils.mask_methods import ensure_masked_bandlimit_mesh_signal
from sleplet.utils.string_methods import filename_args

COEFFICIENTS_TO_NOT_MASK: str = "slepian"


@dataclass
class MeshCoefficients:
    mesh: Mesh
    extra_args: Optional[list[int]] = None
    noise: Optional[float] = None
    region: bool = False

    def __post_init__(self) -> None:
        self._setup_args()
        self._create_name()
        self._create_coefficients()
        self._add_details_to_name()
        self._add_noise_to_signal()

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

    @coefficients.setter
    def coefficients(self, coefficients: np.ndarray) -> None:
        if (
            self.region
            and COEFFICIENTS_TO_NOT_MASK not in self.__class__.__name__.lower()
        ):
            coefficients = ensure_masked_bandlimit_mesh_signal(self.mesh, coefficients)
        self._coefficients = coefficients

    @abstractmethod
    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise to the signal
        """
        raise NotImplementedError

    @abstractmethod
    def _create_coefficients(self) -> None:
        """
        creates the flm on the north pole
        """
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> None:
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
