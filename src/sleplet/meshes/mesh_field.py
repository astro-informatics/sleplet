"""Contains the `MeshField` class."""

from __future__ import annotations

import typing

import igl

if typing.TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._validation
import sleplet.harmonic_methods
import sleplet.meshes.mesh_harmonic_coefficients


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class MeshField(sleplet.meshes.mesh_harmonic_coefficients.MeshHarmonicCoefficients):
    """Create a per-vertex normals field on a given mesh."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex128 | np.float64]:
        """Compute field on the vertices of the mesh."""
        field = igl.per_vertex_normals(self.mesh.vertices, self.mesh.faces)[:, 1]
        return sleplet.harmonic_methods.mesh_forward(self.mesh, field)

    def _create_name(self: typing_extensions.Self) -> str:
        return f"{self.mesh.name}_field"

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            msg = f"{self.__class__.__name__} does not support extra arguments"
            raise TypeError(msg)
