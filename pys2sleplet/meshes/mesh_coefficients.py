from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pys2sleplet.meshes.classes.mesh import Mesh
from pys2sleplet.utils.config import settings
from pys2sleplet.utils.mask_methods import ensure_masked_bandlimit_mesh_signal

COEFFICIENTS_TO_NOT_MASK: set[str] = {"slepian"}


@dataclass  # type:ignore
class MeshCoefficients:
    extra_args: Optional[list[int]]
    noise: Optional[float]
    region: Optional[np.ndarray]
    _coefficients: np.ndarray = field(init=False, repr=False)
    _extra_args: Optional[list[int]] = field(default=None, init=False, repr=False)
    _mesh: Mesh = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _noise: Optional[float] = field(default=None, init=False, repr=False)
    _region: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.mesh = Mesh(self.name, mesh_laplacian=settings.LAPLACIAN)
        self._setup_args()
        self._create_name()
        self._create_coefficients()
        self._add_region_to_name()
        self._add_noise_to_signal()

    def _add_region_to_name(self) -> None:
        """
        adds region to the name if present if not a Slepian function
        """
        if isinstance(self.region, np.ndarray) and "slepian" not in self.name:
            self.name += "_region"

    @property
    def coefficients(self) -> np.ndarray:
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients: np.ndarray) -> None:
        if (
            isinstance(self.region, np.ndarray)
            and not set(self.name.split("_")) & COEFFICIENTS_TO_NOT_MASK
        ):
            coefficients = ensure_masked_bandlimit_mesh_signal(
                self.mesh.basis_functions, self.mesh.region, coefficients
            )
        self._coefficients = coefficients

    @property  # type:ignore
    def extra_args(self) -> Optional[list[int]]:
        return self._extra_args

    @extra_args.setter
    def extra_args(self, extra_args: Optional[list[int]]) -> None:
        if isinstance(extra_args, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            extra_args = MeshCoefficients._extra_args
        self._extra_args = extra_args

    @property
    def mesh(self) -> Mesh:
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        self._mesh = mesh

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property  # type:ignore
    def noise(self) -> Optional[float]:
        return self._noise

    @noise.setter
    def noise(self, noise: Optional[float]) -> None:
        if isinstance(noise, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            noise = MeshCoefficients._noise
        self._noise = noise

    @property  # type:ignore
    def region(self) -> Optional[np.ndarray]:
        return self._region

    @region.setter
    def region(self, region: Optional[np.ndarray]) -> None:
        if isinstance(region, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            region = MeshCoefficients._region
        self._region = region

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
