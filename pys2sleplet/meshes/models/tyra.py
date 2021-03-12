from dataclasses import dataclass

from pys2sleplet.meshes.mesh import Mesh

YMIN = -0.4
YMAX = 0.3
ZMAX = -1.3


@dataclass
class Tyra(Mesh):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_name(self) -> None:
        self.name = "tyra.obj"

    def _setup_region(self) -> None:
        self.region = (
            (self.vertices[1] > YMIN)
            & (self.vertices[1] < YMAX)
            & (self.vertices[2] < ZMAX)
        )
