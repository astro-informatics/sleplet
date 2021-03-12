from dataclasses import dataclass

from pys2sleplet.meshes.mesh import Mesh

XMIN = -0.3
XMAX = 0.3
YMAX = 1.5
ZMAX = -0.5


@dataclass
class Armadillo(Mesh):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_name(self) -> None:
        self.name = "armadillo"

    def _setup_region(self) -> None:
        self.region = (
            (self.vertices[0] > XMIN)
            & (self.vertices[0] < XMAX)
            & (self.vertices[1] < YMAX)
            & (self.vertices[2] < ZMAX)
        )
