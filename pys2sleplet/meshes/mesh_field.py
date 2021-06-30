from dataclasses import dataclass, field

import numpy as np
from igl import principal_curvature

from pys2sleplet.meshes.mesh import Mesh


@dataclass
class MeshField:
    mesh: Mesh
    _function: np.ndarray = field(init=False, repr=False)
    _mesh: Mesh = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compute_function()

    def _compute_function(self) -> None:
        """
        compute field on the vertices of the mesh
        """
        (
            _,
            _,
            self.function,
            _,
        ) = principal_curvature(self.mesh.vertices, self.mesh.faces)

    @property
    def field(self) -> np.ndarray:
        return self._field

    @field.setter
    def field(self, field: np.ndarray) -> None:
        self._field = field

    @property  # type: ignore
    def mesh(self) -> Mesh:
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        self._mesh = mesh
