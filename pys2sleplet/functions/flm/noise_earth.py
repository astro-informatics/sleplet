from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.functions.f_lm import F_LM
from pys2sleplet.functions.flm.earth import Earth
from pys2sleplet.utils.noise import compute_snr, create_noise
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class NoiseEarth(F_LM):
    SNR: float
    _SNR: float = field(default=10, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        earth = Earth(self.L, smoothed=self.smoothed)
        noise = create_noise(self.L, earth.coefficients, self.SNR)
        compute_snr(self.L, earth.coefficients, noise)
        self.coefficients = noise

    def _create_name(self) -> None:
        self.name = f"noise_earth{filename_args(self.SNR, 'snr')}"

    def _set_reality(self) -> None:
        self.reality = True

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
            SNR = NoiseEarth._SNR
        self._SNR = SNR
