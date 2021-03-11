from dataclasses import dataclass

from pys2sleplet.meshes.mesh import Mesh


@dataclass
class Bunny(Mesh):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_name(self) -> None:
        self.name = "bunny"
