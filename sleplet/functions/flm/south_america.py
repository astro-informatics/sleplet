from pathlib import Path

import numpy as np
import pyssht as ssht
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet.data.other.earth.create_earth_flm import create_flm
from sleplet.functions.f_lm import F_LM
from sleplet.utils.harmonic_methods import ensure_f_bandlimited
from sleplet.utils.plot_methods import rotate_earth_to_south_america
from sleplet.utils.string_methods import convert_camel_case_to_snake_case
from sleplet.utils.validation import Validation
from sleplet.utils.vars import SAMPLING_SCHEME

_file_location = Path(__file__).resolve()
_mask_path = _file_location.parents[2] / "data" / "slepian" / "masks"


@dataclass(config=Validation)
class SouthAmerica(F_LM):
    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_]:
        return ensure_f_bandlimited(self._grid_fun, self.L, self.reality, self.spin)

    def _create_name(self) -> str:
        return convert_camel_case_to_snake_case(self.__class__.__name__)

    def _set_reality(self) -> bool:
        return True

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )

    def _grid_fun(
        self, theta: npt.NDArray[np.float_], phi: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """
        function on the grid
        """
        earth_flm = create_flm(self.L, smoothing=self.smoothing)
        rot_flm = rotate_earth_to_south_america(earth_flm, self.L)
        earth_f = ssht.inverse(
            rot_flm, self.L, Reality=self.reality, Method=SAMPLING_SCHEME
        )
        mask = np.load(_mask_path / f"{self.name}_L{self.L}.npy")
        return np.where(mask, earth_f, 0)
