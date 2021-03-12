from abc import abstractmethod
from dataclasses import dataclass

from pys2sleplet.meshes.mesh import Mesh

XMIN = -0.7
YMIN = 0.1
YMAX = 1.4
ZMAX = -0.4


@dataclass  # type: ignore
class Bunny(Mesh):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _setup_region(self) -> None:
        self.region = (
            (self.vertices[0] > XMIN)
            & (self.vertices[1] > YMIN)
            & (self.vertices[1] < YMAX)
            & (self.vertices[2] < ZMAX)
        )

    @abstractmethod
    def _create_name(self) -> None:
        raise NotImplementedError
