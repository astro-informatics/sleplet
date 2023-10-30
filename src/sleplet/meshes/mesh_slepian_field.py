"""Contains the `MeshSlepianField` class."""
import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._validation
import sleplet.meshes.mesh_field
import sleplet.slepian_methods
from sleplet.meshes.mesh_slepian_coefficients import MeshSlepianCoefficients


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class MeshSlepianField(MeshSlepianCoefficients):
    """
    Create a field on a given mesh computed from a Slepian region of the mesh.
    The default field is the per-vertex normals of the mesh.
    """

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        """Compute field on the vertices of the mesh."""
        mf = sleplet.meshes.mesh_field.MeshField(
            self.mesh,
            region=True,
        )
        return sleplet.slepian_methods.slepian_mesh_forward(
            self.mesh_slepian,
            u_i=mf.coefficients,
        )

    def _create_name(self: typing_extensions.Self) -> str:
        return f"slepian_{self.mesh.name}_field"

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            msg = f"{self.__class__.__name__} does not support extra arguments"
            raise TypeError(msg)
