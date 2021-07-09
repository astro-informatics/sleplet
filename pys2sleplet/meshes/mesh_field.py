from dataclasses import dataclass, field

import numpy as np
from igl import principal_curvature

from pys2sleplet.meshes.mesh import Mesh
from pys2sleplet.utils.mesh_methods import average_functions_on_vertices_to_faces


@dataclass
class MeshField:
    mesh: Mesh
    mask_region: bool
    _field_values: np.ndarray = field(init=False, repr=False)
    _mesh: Mesh = field(init=False, repr=False)
    _mask_region: bool = field(default=False, init=False, repr=False)

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
        self._set_outside_region_to_zero()

    def _set_outside_region_to_zero(self) -> None:
        """
        method to set the outside the region to zero
        """
        if self.mask_region:
            self.field_values = np.where(self.mesh.region, self.field_values, 0)

    @property
    def field_values(self) -> np.ndarray:
        return self._field_values

    @field_values.setter
    def field_values(self, field_values: np.ndarray) -> None:
        self._field_values = field_values

    @property  # type: ignore
    def mask_region(self) -> bool:
        return self._mask_region

    @mask_region.setter
    def mask_region(self, mask_region: bool) -> None:
        if isinstance(mask_region, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            mask_region = MeshField._mask_region
        self._mask_region = mask_region

    @property  # type: ignore
    def mesh(self) -> Mesh:
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        self._mesh = mesh
