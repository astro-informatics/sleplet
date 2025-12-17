"""Contains the abstract `MeshSlepianCoefficients` class."""

from __future__ import annotations

import abc
import typing

if typing.TYPE_CHECKING:
    import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._validation
import sleplet.meshes.mesh_coefficients
import sleplet.meshes.mesh_slepian
import sleplet.noise


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class MeshSlepianCoefficients(sleplet.meshes.mesh_coefficients.MeshCoefficients):
    """Abstract parent class to handle Slepian coefficients on the mesh."""

    mesh_slepian: sleplet.meshes.mesh_slepian.MeshSlepian | None = pydantic.Field(
        default=None,
        init_var=False,
        repr=False,
    )

    def __post_init__(self: typing_extensions.Self) -> None:
        self.mesh_slepian = sleplet.meshes.mesh_slepian.MeshSlepian(self.mesh)
        super().__post_init__()

    def _add_noise_to_signal(
        self: typing_extensions.Self,
    ) -> tuple[npt.NDArray[np.complex128 | np.float64] | None, float | None]:
        """Add Gaussian white noise converted to Slepian space."""
        self.coefficients: npt.NDArray[np.complex128 | np.float64]
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

    @abc.abstractmethod
    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex128 | np.float64]:
        raise NotImplementedError

    @abc.abstractmethod
    def _create_name(self: typing_extensions.Self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def _setup_args(self: typing_extensions.Self) -> None:
        raise NotImplementedError
