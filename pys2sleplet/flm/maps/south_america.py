from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyssht as ssht

from pys2sleplet.data.other.earth.create_earth_flm import create_flm
from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.harmonic_methods import ensure_f_bandlimited
from pys2sleplet.utils.vars import (
    EARTH_ALPHA,
    EARTH_BETA,
    EARTH_GAMMA,
    SOUTH_AMERICA_RANGE,
)

_file_location = Path(__file__).resolve()


@dataclass
class SouthAmerica(Functions):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        self.multipole = ensure_f_bandlimited(
            self._grid_fun, self.L, self.reality, self.spin
        )

    def _create_name(self) -> None:
        self.name = "south_america"

    def _set_reality(self) -> None:
        self.reality = True

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )

    def _grid_fun(self, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """
        function on the grid
        """
        earth_flm = create_flm(self.L)
        rot_flm = ssht.rotate_flms(
            earth_flm, EARTH_ALPHA, EARTH_BETA, EARTH_GAMMA, self.L
        )
        earth_f = ssht.inverse(rot_flm, self.L, Reality=self.reality)
        mask = (theta <= SOUTH_AMERICA_RANGE) & (earth_f >= 0)
        return np.where(mask, earth_f, 0)
