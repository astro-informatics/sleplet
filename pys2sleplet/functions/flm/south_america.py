from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyssht as ssht

from pys2sleplet.data.other.earth.create_earth_flm import create_flm
from pys2sleplet.functions.f_lm import F_LM
from pys2sleplet.utils.harmonic_methods import ensure_f_bandlimited
from pys2sleplet.utils.vars import EARTH_ALPHA, EARTH_BETA, EARTH_GAMMA, SAMPLING_SCHEME

_file_location = Path(__file__).resolve()
_mask_path = _file_location.parents[2] / "data" / "slepian" / "masks"


@dataclass
class SouthAmerica(F_LM):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        self.coefficients = ensure_f_bandlimited(
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
        earth_f = ssht.inverse(
            rot_flm, self.L, Reality=self.reality, Method=SAMPLING_SCHEME
        )
        mask = np.load(_mask_path / f"{self.name}_L{self.L}.npy")
        return np.where(mask, earth_f, 0)
