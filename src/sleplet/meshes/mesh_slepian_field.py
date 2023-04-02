"""Contains the `MeshSlepianField` class."""
import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._validation
import sleplet.meshes.mesh_field
import sleplet.slepian_methods
from sleplet.meshes.mesh_slepian_coefficients import MeshSlepianCoefficients


@dataclass(config=sleplet._validation.Validation)
class MeshSlepianField(MeshSlepianCoefficients):
    """
    Creates a field on a given mesh computed from a Slepian region of the mesh.
    The default field is the per-vertex normals of the mesh.
    """

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        """Compute field on the vertices of the mesh."""
        mf = sleplet.meshes.mesh_field.MeshField(
            self.mesh,
            region=True,
        )
        return sleplet.slepian_methods.slepian_mesh_forward(
            self.mesh_slepian,
            u_i=mf.coefficients,
        )

    def _create_name(self) -> str:
        return f"slepian_{self.mesh.name}_field"

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments",
            )
