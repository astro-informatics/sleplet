from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyssht as ssht

from pys2sleplet.data.maps.earth.create_earth_flm import create_flm
from pys2sleplet.flm.functions import Functions
from pys2sleplet.utils.vars import EARTH_ALPHA, EARTH_BETA, EARTH_GAMMA, SAMPLING_SCHEME

_file_location = Path(__file__).resolve()


@dataclass
class SouthAmerica(Functions):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        earth_flm = create_flm(self.L)
        rot_flm = ssht.rotate_flms(
            earth_flm, EARTH_ALPHA, EARTH_BETA, EARTH_GAMMA, self.L
        )
        earth_f = ssht.inverse(
            rot_flm, self.L, Method=SAMPLING_SCHEME, Reality=self.reality
        )
        theta_grid, _ = ssht.sample_positions(self.L, Grid=True, Method=SAMPLING_SCHEME)
        mask = (theta_grid <= np.deg2rad(40)) & (earth_f >= 0)
        masked_f = np.where(mask, earth_f, 0)
        self.multipole = ssht.forward(
            masked_f, self.L, Method="MWSS", Reality=self.reality
        )

    def _create_name(self) -> None:
        self.name = "south_america"

    def _set_reality(self) -> None:
        self.reality = True

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
