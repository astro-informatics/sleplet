from dataclasses import field

import numpy as np
from igl import per_vertex_normals
from pydantic.dataclasses import dataclass

from sleplet.meshes.mesh_harmonic_coefficients import MeshHarmonicCoefficients
from sleplet.utils.harmonic_methods import mesh_forward
from sleplet.utils.validation import Validation


@dataclass(config=Validation, kw_only=True)
class MeshField(MeshHarmonicCoefficients):
    coefficients: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        """
        compute field on the vertices of the mesh
        """
        field = per_vertex_normals(self.mesh.vertices, self.mesh.faces)[:, 1]
        self.coefficients = mesh_forward(self.mesh, field)

    def _create_name(self) -> None:
        self.name = f"{self.mesh.name}_field"

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
