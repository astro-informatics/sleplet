from dataclasses import dataclass

from pys2sleplet.data.other.earth.create_earth_flm import create_flm
from pys2sleplet.functions.f_lm import F_LM


@dataclass
class Earth(F_LM):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        self.coefficients = create_flm(self.L, smoothed=self.smoothed)

    def _create_name(self) -> None:
        self.name = "earth"

    def _set_reality(self) -> None:
        self.reality = True

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
