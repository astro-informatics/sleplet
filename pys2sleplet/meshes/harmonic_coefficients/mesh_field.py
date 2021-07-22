from dataclasses import dataclass

from igl import principal_curvature

from pys2sleplet.meshes.mesh_harmonic_coefficients import MeshHarmonicCoefficients
from pys2sleplet.utils.harmonic_methods import mesh_forward


@dataclass
class MeshField(MeshHarmonicCoefficients):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        """
        compute field on the vertices of the mesh
        """
        _, _, maximal_curvature_value, _ = principal_curvature(
            self.mesh.vertices, self.mesh.faces
        )
        self.coefficients = mesh_forward(self.mesh, maximal_curvature_value)

    def _create_name(self) -> None:
        self.name = f"{self.mesh.name}_field"

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
