"""Contains the `MeshField` class."""
import igl
import numpy as np
import numpy.typing as npt
import pydantic.v1 as pydantic

import sleplet._validation
import sleplet.harmonic_methods
from sleplet.meshes.mesh_harmonic_coefficients import MeshHarmonicCoefficients


@pydantic.dataclasses.dataclass(config=sleplet._validation.Validation)
class MeshField(MeshHarmonicCoefficients):
    """Creates a per-vertex normals field on a given mesh."""

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        """Compute field on the vertices of the mesh."""
        field = igl.per_vertex_normals(self.mesh.vertices, self.mesh.faces)[:, 1]
        return sleplet.harmonic_methods.mesh_forward(self.mesh, field)

    def _create_name(self) -> str:
        return f"{self.mesh.name}_field"

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments",
            )
