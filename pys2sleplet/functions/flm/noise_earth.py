from dataclasses import dataclass, field

from sleplet.functions.f_lm import F_LM
from sleplet.functions.flm.earth import Earth
from sleplet.utils.noise import compute_snr, create_noise
from sleplet.utils.string_methods import convert_camel_case_to_snake_case, filename_args


@dataclass
class NoiseEarth(F_LM):
    SNR: float
    _SNR: float = field(default=10, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        earth = Earth(self.L, smoothing=self.smoothing)
        noise = create_noise(self.L, earth.coefficients, self.SNR)
        compute_snr(earth.coefficients, noise, "Harmonic")
        self.coefficients = noise

    def _create_name(self) -> None:
        self.name = (
            f"{convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{filename_args(self.SNR, 'snr')}"
        )

    def _set_reality(self) -> None:
        self.reality = True

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.SNR = self.extra_args[0]

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
