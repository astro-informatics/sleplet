import numpy as np
from pydantic.dataclasses import dataclass

from sleplet.meshes.harmonic_coefficients.mesh_field import MeshField
from sleplet.meshes.mesh_slepian_coefficients import MeshSlepianCoefficients
from sleplet.utils.slepian_methods import slepian_mesh_forward
from sleplet.utils.validation import Validation


@dataclass(config=Validation)
class MeshSlepianField(MeshSlepianCoefficients):
    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> np.ndarray:
        """
        compute field on the vertices of the mesh
        """
        mf = MeshField(self.mesh, region=True)
        return slepian_mesh_forward(
            self.mesh_slepian,
            u_i=mf.coefficients,
        )

    def _create_name(self) -> str:
        return f"slepian_{self.mesh.name}_field"

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
