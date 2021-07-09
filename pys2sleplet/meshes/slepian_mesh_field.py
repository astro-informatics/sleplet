from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.meshes.mesh_field import MeshField
from pys2sleplet.meshes.slepian_mesh import SlepianMesh
from pys2sleplet.utils.slepian_mesh_methods import slepian_mesh_forward


@dataclass
class SlepianMeshField:
    mesh_field: MeshField
    slepian_mesh: SlepianMesh
    _mesh_field: MeshField = field(init=False, repr=False)
    _slepian_coefficients: np.ndarray = field(init=False, repr=False)
    _slepian_mesh: SlepianMesh = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compute_slepian_coefficients()

    def _compute_slepian_coefficients(self) -> None:
        """
        computes the Slepian coefficients of the mesh field values
        """
        self.slepian_coefficients = slepian_mesh_forward(
            self.slepian_mesh.mesh,
            self.slepian_mesh.slepian_eigenvalues,
            self.slepian_mesh.slepian_functions,
            self.slepian_mesh.N,
            u=self.mesh_field.field_values,
        )

    @property  # type: ignore
    def mesh_field(self) -> MeshField:
        return self._mesh_field

    @mesh_field.setter
    def mesh_field(self, mesh_field: MeshField) -> None:
        self._mesh_field = mesh_field

    @property
    def slepian_coefficients(self) -> np.ndarray:
        return self._slepian_coefficients

    @slepian_coefficients.setter
    def slepian_coefficients(self, slepian_coefficients: np.ndarray) -> None:
        self._slepian_coefficients = slepian_coefficients

    @property  # type: ignore
    def slepian_mesh(self) -> SlepianMesh:
        return self._slepian_mesh

    @slepian_mesh.setter
    def slepian_mesh(self, slepian_mesh: SlepianMesh) -> None:
        self._slepian_mesh = slepian_mesh
