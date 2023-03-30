import numpy as np
from igl import per_vertex_normals
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._validation
import sleplet.harmonic_methods
import sleplet.meshes.mesh_harmonic_coefficients


@dataclass(config=sleplet._validation.Validation)
class MeshField(sleplet.meshes.mesh_harmonic_coefficients.MeshHarmonicCoefficients):
    """TODO"""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        """
        compute field on the vertices of the mesh
        """
        field = per_vertex_normals(self.mesh.vertices, self.mesh.faces)[:, 1]
        return sleplet.harmonic_methods.mesh_forward(self.mesh, field)

    def _create_name(self) -> str:
        return f"{self.mesh.name}_field"

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
