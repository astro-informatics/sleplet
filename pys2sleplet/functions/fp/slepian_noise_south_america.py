from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.functions.f_p import F_P
from pys2sleplet.functions.fp.slepian_south_america import SlepianSouthAmerica
from pys2sleplet.utils.noise import compute_snr, create_slepian_noise
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class SlepianNoiseSouthAmerica(F_P):
    SNR: float
    _SNR: float = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.region.name_ending != "south_america":
            raise RuntimeError("Slepian region selected must be 'south_america'")

    def _create_coefficients(self) -> None:
        sa = SlepianSouthAmerica(self.L, region=self.region, smoothing=self.smoothing)
        noise = create_slepian_noise(self.L, sa.coefficients, self.slepian, self.SNR)
        compute_snr(self.L, sa.coefficients, noise)
        self.coefficients = noise

    def _create_name(self) -> None:
        self.name = f"slepian_noise_south_america{filename_args(self.SNR, 'snr')}"

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.SNR = np.float_power(10, self.extra_args[0])

    @property  # type:ignore
    def SNR(self) -> float:
        return self._SNR

    @SNR.setter
    def SNR(self, SNR: float) -> None:
        if isinstance(SNR, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            SNR = SlepianNoiseSouthAmerica._SNR
        self._SNR = SNR
