"""Contains the `MeshSlepianNoiseField` class."""
import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._string_methods
import sleplet._validation
import sleplet.meshes.mesh_slepian_field
import sleplet.noise
from sleplet.meshes.mesh_slepian_coefficients import MeshSlepianCoefficients


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class MeshSlepianNoiseField(MeshSlepianCoefficients):
    """
    Creates a noisedfield on a given mesh computed from a Slepian region of the
    mesh. The default field is the per-vertex normals of the mesh.
    """

    SNR: float = -5
    """A parameter which controls the level of signal-to-noise in the noised
    data."""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        smf = sleplet.meshes.mesh_slepian_field.MeshSlepianField(
            self.mesh,
            region=True,
        )
        noise = sleplet.noise._create_slepian_mesh_noise(
            self.mesh_slepian,
            smf.coefficients,
            self.SNR,
        )
        sleplet.noise.compute_snr(smf.coefficients, noise, "Slepian")
        return noise

    def _create_name(self) -> str:
        return (
            f"slepian_{self.mesh.name}_noise_field"
            f"{sleplet._string_methods.filename_args(self.SNR, 'snr')}"
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.SNR = self.extra_args[0]
