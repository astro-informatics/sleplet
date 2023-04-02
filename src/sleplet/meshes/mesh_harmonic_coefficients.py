"""Contains the abstract `MeshHarmonicCoefficients` class."""
from abc import abstractmethod

import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._validation
import sleplet.noise
from sleplet.meshes.mesh_coefficients import MeshCoefficients


@dataclass(config=sleplet._validation.Validation)
class MeshHarmonicCoefficients(MeshCoefficients):
    """Abstract parent class to handle Fourier coefficients on the mesh."""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _add_noise_to_signal(
        self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """Adds Gaussian white noise to the signal."""
        self.coefficients: npt.NDArray[np.complex_ | np.float_]
        if self.noise is not None:
            unnoised_coefficients = self.coefficients.copy()
            nlm = sleplet.noise._create_mesh_noise(self.coefficients, self.noise)
            snr = sleplet.noise.compute_snr(self.coefficients, nlm, "Harmonic")
            self.coefficients = self.coefficients + nlm
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
