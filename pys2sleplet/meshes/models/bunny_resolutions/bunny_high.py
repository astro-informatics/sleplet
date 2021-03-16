from dataclasses import dataclass

from pys2sleplet.meshes.models.bunny import Bunny

XMIN = -0.05
YMIN = 0.05
YMAX = 0.12
ZMAX = -0.016


@dataclass  # type: ignore
class BunnyHigh(Bunny):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_name(self) -> None:
        self.name = "bunny_3485.off"

    def _setup_region(self) -> None:
        self.region = (
            (self.vertices[:, 0] > XMIN)
            & (self.vertices[:, 1] > YMIN)
            & (self.vertices[:, 1] < YMAX)
            & (self.vertices[:, 2] < ZMAX)
        )
