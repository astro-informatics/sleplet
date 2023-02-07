from abc import abstractmethod
from dataclasses import KW_ONLY, field
from typing import Optional

import numpy as np
from pydantic import validator
from pydantic.dataclasses import dataclass

from sleplet.meshes.classes.mesh import Mesh
from sleplet.meshes.classes.mesh_slepian import MeshSlepian
from sleplet.utils.mask_methods import ensure_masked_bandlimit_mesh_signal
from sleplet.utils.string_methods import filename_args
from sleplet.utils.validation import Validation

COEFFICIENTS_TO_NOT_MASK: str = "slepian"


@dataclass(config=Validation)
class MeshCoefficients:
    mesh: Mesh
    _: KW_ONLY
    coefficients: np.ndarray = field(init=False, repr=False)
    mesh_slepian: MeshSlepian = field(init=False, repr=False)
    name: str = field(init=False, repr=False)
    unnoised_coefficients: np.ndarray = field(
        default=np.array([0]), init=False, repr=False
    )
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

    @validator("coefficients")
    def check_coefficients(cls, v, values):
        if (
            "region" in values
            and COEFFICIENTS_TO_NOT_MASK not in cls.__class__.__name__.lower()
        ):
            v = ensure_masked_bandlimit_mesh_signal(values["mesh"], v)
        return v

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
