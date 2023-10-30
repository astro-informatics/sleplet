"""Contains the `MeshNoiseField` class."""
import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._string_methods
import sleplet._validation
import sleplet.meshes.mesh_slepian
import sleplet.noise
from sleplet.meshes.mesh_harmonic_coefficients import MeshHarmonicCoefficients


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class MeshNoiseField(MeshHarmonicCoefficients):
    """Create a noised per-vertex normals field on a given mesh."""

    SNR: float = 10
    """A parameter which controls the level of signal-to-noise in the noised
    data."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        mf = sleplet.meshes.mesh_slepian.MeshField(self.mesh)
        noise = sleplet.noise._create_mesh_noise(mf.coefficients, self.SNR)
        sleplet.noise.compute_snr(mf.coefficients, noise, "Harmonic")
        return noise

    def _create_name(self: typing_extensions.Self) -> str:
        return (
            f"{self.mesh.name}_noise_field"
            f"{sleplet._string_methods.filename_args(self.SNR, 'snr')}"
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return True

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                msg = f"The number of extra arguments should be {num_args}"
                raise ValueError(msg)
            self.SNR = self.extra_args[0]
