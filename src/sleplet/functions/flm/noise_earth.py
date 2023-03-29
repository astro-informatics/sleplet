import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet._string_methods import (
    _convert_camel_case_to_snake_case,
    filename_args,
)
from sleplet._validation import Validation
from sleplet.functions.f_lm import F_LM
from sleplet.functions.flm.earth import Earth
from sleplet.noise import _create_noise, compute_snr


@dataclass(config=Validation, kw_only=True)
class NoiseEarth(F_LM):
    SNR: float = 10

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        earth = Earth(self.L, smoothing=self.smoothing)
        noise = _create_noise(self.L, earth.coefficients, self.SNR)
        compute_snr(earth.coefficients, noise, "Harmonic")
        return noise

    def _create_name(self) -> str:
        return (
            f"{_convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{filename_args(self.SNR, 'snr')}"
        )

    def _set_reality(self) -> bool:
        return True

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.SNR = self.extra_args[0]
