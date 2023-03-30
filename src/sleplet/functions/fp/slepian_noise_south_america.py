import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._string_methods
import sleplet._validation
import sleplet.functions.f_p
import sleplet.functions.fp.slepian_south_america
import sleplet.noise
from sleplet.slepian.region import Region


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class SlepianNoiseSouthAmerica(sleplet.functions.f_p.F_P):
    """TODO"""

    SNR: float = -10
    """TODO"""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()
        if (
            isinstance(self.region, Region)
            and self.region.name_ending != "south_america"
        ):
            raise RuntimeError("Slepian region selected must be 'south_america'")

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        sa = sleplet.functions.fp.slepian_south_america.SlepianSouthAmerica(
            self.L, region=self.region, smoothing=self.smoothing
        )
        noise = sleplet.noise._create_slepian_noise(
            self.L, sa.coefficients, self.slepian, self.SNR
        )
        sleplet.noise.compute_snr(sa.coefficients, noise, "Slepian")
        return noise

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.SNR, 'snr')}"
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.SNR = self.extra_args[0]
