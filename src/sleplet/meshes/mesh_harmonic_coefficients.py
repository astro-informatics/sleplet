"""Contains the abstract `MeshHarmonicCoefficients` class."""
import abc

import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._validation
import sleplet.noise
from sleplet.meshes.mesh_coefficients import MeshCoefficients


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class MeshHarmonicCoefficients(MeshCoefficients):
    """Abstract parent class to handle Fourier coefficients on the mesh."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _add_noise_to_signal(
        self: typing_extensions.Self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """Add Gaussian white noise to the signal."""
        self.coefficients: npt.NDArray[np.complex_ | np.float_]
        if self.noise is not None:
            unnoised_coefficients = self.coefficients.copy()
            nlm = sleplet.noise._create_mesh_noise(self.coefficients, self.noise)
            snr = sleplet.noise.compute_snr(self.coefficients, nlm, "Harmonic")
            self.coefficients = self.coefficients + nlm
            return unnoised_coefficients, snr
        return None, None

    @abc.abstractmethod
    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        raise NotImplementedError

    @abc.abstractmethod
    def _create_name(self: typing_extensions.Self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_args(self: typing_extensions.Self) -> None:
        raise NotImplementedError
