from dataclasses import dataclass

from pys2sleplet.meshes.mesh import Mesh

ZMAX = -2.7


@dataclass
class Bumpy(Mesh):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_name(self) -> None:
        self.name = "bumpy.off"

    def _setup_region(self) -> None:
        self.region = self.vertices[:, 2] < ZMAX
