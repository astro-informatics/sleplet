import numpy as np
from pydantic.dataclasses import dataclass

from sleplet.functions.f_p import F_P
from sleplet.functions.fp.slepian_africa import SlepianAfrica
from sleplet.utils.noise import compute_snr, create_slepian_noise
from sleplet.utils.region import Region
from sleplet.utils.string_methods import convert_camel_case_to_snake_case, filename_args
from sleplet.utils.validation import Validation


@dataclass(config=Validation, kw_only=True)
class SlepianNoiseAfrica(F_P):
    SNR: float = -10

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.region, Region) and self.region.name_ending != "africa":
            raise RuntimeError("Slepian region selected must be 'africa'")

    def _create_coefficients(self) -> np.ndarray:
        sa = SlepianAfrica(self.L, region=self.region, smoothing=self.smoothing)
        noise = create_slepian_noise(self.L, sa.coefficients, self.slepian, self.SNR)
        compute_snr(sa.coefficients, noise, "Slepian")
        return noise

    def _create_name(self) -> str:
        return (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{filename_args(self.SNR, 'snr')}"
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
