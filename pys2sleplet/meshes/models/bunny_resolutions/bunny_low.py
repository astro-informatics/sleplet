from dataclasses import dataclass

from pys2sleplet.meshes.models.bunny import Bunny


@dataclass  # type: ignore
class BunnyLow(Bunny):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_name(self) -> None:
        self.name = "bunny_453.ply"
