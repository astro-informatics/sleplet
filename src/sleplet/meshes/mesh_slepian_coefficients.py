"""Contains the abstract `MeshSlepianCoefficients` class."""
from abc import abstractmethod

import numpy as np
import pydantic
from numpy import typing as npt

import sleplet._validation
import sleplet.meshes.mesh_slepian
import sleplet.noise
from sleplet.meshes.mesh_coefficients import MeshCoefficients


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class MeshSlepianCoefficients(MeshCoefficients):
    """Abstract parent class to handle Slepian coefficients on the mesh."""

    def __post_init__(self) -> None:
        self.mesh_slepian = sleplet.meshes.mesh_slepian.MeshSlepian(self.mesh)
        super().__post_init__()

    def _add_noise_to_signal(
        self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """Adds Gaussian white noise converted to Slepian space."""
        self.coefficients: npt.NDArray[np.complex_ | np.float_]
        if self.noise is not None:
            unnoised_coefficients = self.coefficients.copy()
            n_p = sleplet.noise._create_slepian_mesh_noise(
                self.mesh_slepian,
                self.coefficients,
                self.noise,
            )
            snr = sleplet.noise.compute_snr(self.coefficients, n_p, "Slepian")
            self.coefficients = self.coefficients + n_p
            return unnoised_coefficients, snr
        return None, None

    @abstractmethod
    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        raise NotImplementedError
