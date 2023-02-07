from pydantic.dataclasses import dataclass

from sleplet.functions.f_p import F_P
from sleplet.functions.fp.slepian_south_america import SlepianSouthAmerica
from sleplet.utils.noise import compute_snr, create_slepian_noise
from sleplet.utils.region import Region
from sleplet.utils.string_methods import convert_camel_case_to_snake_case, filename_args
from sleplet.utils.validation import Validation


@dataclass(config=Validation, kw_only=True)
class SlepianNoiseSouthAmerica(F_P):
    int: float = -10

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            isinstance(self.region, Region)
            and self.region.name_ending != "south_america"
        ):
            raise RuntimeError("Slepian region selected must be 'south_america'")

    def _create_coefficients(self) -> None:
        sa = SlepianSouthAmerica(self.L, region=self.region, smoothing=self.smoothing)
        noise = create_slepian_noise(self.L, sa.coefficients, self.slepian, self.SNR)
        compute_snr(sa.coefficients, noise, "Slepian")
        self.coefficients = noise

    def _create_name(self) -> None:
        self.name = (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{filename_args(self.SNR, 'snr')}"
        )

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.SNR = self.extra_args[0]
