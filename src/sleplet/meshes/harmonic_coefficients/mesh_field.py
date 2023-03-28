import numpy as np
from igl import per_vertex_normals
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet.meshes.mesh_harmonic_coefficients import MeshHarmonicCoefficients
from sleplet.utils._validation import Validation
from sleplet.utils.harmonic_methods import _mesh_forward


@dataclass(config=Validation)
class MeshField(MeshHarmonicCoefficients):
    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        """
        compute field on the vertices of the mesh
        """
        field = per_vertex_normals(self.mesh.vertices, self.mesh.faces)[:, 1]
        return _mesh_forward(self.mesh, field)

    def _create_name(self) -> str:
        return f"{self.mesh.name}_field"

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
