from dataclasses import dataclass

from sleplet.meshes.harmonic_coefficients.mesh_field import MeshField
from sleplet.meshes.mesh_slepian_coefficients import MeshSlepianCoefficients
from sleplet.utils.slepian_methods import slepian_mesh_forward


@dataclass
class MeshSlepianField(MeshSlepianCoefficients):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        """
        compute field on the vertices of the mesh
        """
        mf = MeshField(self.mesh, region=True)
        self.coefficients = slepian_mesh_forward(
            self.mesh_slepian,
            u_i=mf.coefficients,
        )

    def _create_name(self) -> None:
        self.name = f"slepian_{self.mesh.name}_field"

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
