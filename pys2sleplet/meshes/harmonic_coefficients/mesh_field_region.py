from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.meshes.harmonic_coefficients.mesh_field import MeshField
from pys2sleplet.utils.mesh_methods import bandlimit_signal


@dataclass
class MeshFieldRegion:
    mesh_field: MeshField
    _field_values: np.ndarray = field(init=False, repr=False)
    _mesh_field: MeshField = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._compute_field_values()

    def _compute_field_values(self) -> None:
        """
        set the field values outside the region to zero
        """
        masked_region = np.where(
            self.mesh_field.mesh.region, self.mesh_field.field_values, 0
        )
        self.field_values = bandlimit_signal(
            self.mesh_field.mesh.basis_functions,
            masked_region,
        )

    @property
    def field_values(self) -> np.ndarray:
        return self._field_values

    @field_values.setter
    def field_values(self, field_values: np.ndarray) -> None:
        self._field_values = field_values

    @property  # type: ignore
    def mesh_field(self) -> MeshField:
        return self._mesh_field

    @mesh_field.setter
    def mesh_field(self, mesh_field: MeshField) -> None:
        self._mesh_field = mesh_field
