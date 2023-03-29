import numpy as np
import pyssht as ssht
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet._mask_methods import create_mask
from sleplet._validation import Validation
from sleplet._vars import SAMPLING_SCHEME
from sleplet.data.create_earth_flm import create_flm
from sleplet.data.setup_pooch import find_on_pooch_then_local
from sleplet.functions.f_lm import F_LM
from sleplet.harmonic_methods import _ensure_f_bandlimited, rotate_earth_to_africa
from sleplet.string_methods import _convert_camel_case_to_snake_case


@dataclass(config=Validation)
class Africa(F_LM):
    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        return _ensure_f_bandlimited(
            self._grid_fun, self.L, reality=self.reality, spin=self.spin
        )

    def _create_name(self) -> str:
        return _convert_camel_case_to_snake_case(self.__class__.__name__)

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
        rot_flm = rotate_earth_to_africa(earth_flm, self.L)
        earth_f = ssht.inverse(
            rot_flm, self.L, Reality=self.reality, Method=SAMPLING_SCHEME
        )
        mask_name = f"{self.name}_L{self.L}.npy"
        mask_location = find_on_pooch_then_local(f"slepian_masks_{mask_name}")
        mask = (
            create_mask(self.L, mask_name)
            if mask_location is None
            else np.load(mask_location)
        )
        return np.where(mask, earth_f, 0)
