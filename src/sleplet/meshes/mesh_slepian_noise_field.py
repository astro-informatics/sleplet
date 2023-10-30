"""Contains the `MeshSlepianNoiseField` class."""
import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._string_methods
import sleplet._validation
import sleplet.meshes.mesh_slepian_field
import sleplet.noise
from sleplet.meshes.mesh_slepian_coefficients import MeshSlepianCoefficients


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class MeshSlepianNoiseField(MeshSlepianCoefficients):
    """
    Create a noisedfield on a given mesh computed from a Slepian region of the
    mesh. The default field is the per-vertex normals of the mesh.
    """

    SNR: float = -5
    """A parameter which controls the level of signal-to-noise in the noised
    data."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
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

    def _create_name(self: typing_extensions.Self) -> str:
        return (
            f"slepian_{self.mesh.name}_noise_field"
            f"{sleplet._string_methods.filename_args(self.SNR, 'snr')}"
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return False

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                msg = f"The number of extra arguments should be {num_args}"
                raise ValueError(msg)
            self.SNR = self.extra_args[0]
