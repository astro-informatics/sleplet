from dataclasses import dataclass
from pathlib import Path

from pys2sleplet.data.maps.earth.create_earth_flm import create_flm
from pys2sleplet.flm.functions import Functions

_file_location = Path(__file__).resolve()


@dataclass
class Earth(Functions):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        self.multipole = create_flm(self.L)

    def _create_name(self) -> None:
        self.name = "earth"

    def _set_reality(self) -> None:
        self.reality = True

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
