from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from igl import principal_curvature

from pys2sleplet.meshes.mesh import Mesh
from pys2sleplet.utils.mesh_methods import (
    add_noise_to_mesh,
    average_functions_on_vertices_to_faces,
)


@dataclass
class MeshField:
    mesh: Mesh
    noise: Optional[int]
    _field_values: np.ndarray = field(init=False, repr=False)
    _mesh: Mesh = field(init=False, repr=False)
    _noise: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._compute_field_values()

    def _compute_field_values(self) -> None:
        """
        compute field on the vertices of the mesh
        """
        _, _, maximal_curvature, _ = principal_curvature(
            self.mesh.vertices, self.mesh.faces
        )
        self.field_values = average_functions_on_vertices_to_faces(
            self.mesh.faces, maximal_curvature
        )
        if self.noise is not None:
            self.field_values, _ = add_noise_to_mesh(
                self.mesh.vertices,
                self.mesh.faces,
                self.mesh.basis_functions,
                self.field_values,
                self.noise,
            )

    @property
    def field_values(self) -> np.ndarray:
        return self._field_values

    @field_values.setter
    def field_values(self, field_values: np.ndarray) -> None:
        self._field_values = field_values

    @property  # type: ignore
    def mesh(self) -> Mesh:
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        self._mesh = mesh

    @property  # type: ignore
    def noise(self) -> Optional[int]:
        return self._noise

    @noise.setter
    def noise(self, noise: Optional[int]) -> None:
        if isinstance(noise, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            noise = MeshField._noise
        self._noise = noise
