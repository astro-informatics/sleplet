from dataclasses import dataclass, field

from pys2sleplet.functions.f_p import F_P
from pys2sleplet.functions.flm.south_america import SouthAmerica
from pys2sleplet.utils.noise import create_noise
from pys2sleplet.utils.slepian_methods import slepian_forward
from pys2sleplet.utils.string_methods import filename_args


@dataclass
class SlepianNoiseGaussianSouthAmerica(F_P):
    SNR: float
    _SNR: float = field(default=10, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.region.name_ending != "south_america":
            raise RuntimeError("Slepian region selected must be 'south_america'")

    def _create_annotations(self) -> None:
        self.annotations = self.slepian.annotations

    def _create_coefficients(self) -> None:
        sa = SouthAmerica(self.L, region=self.region)
        harmonic_noise = create_noise(self.L, sa.coefficients, self.SNR)
        self.coefficients = slepian_forward(self.L, harmonic_noise, self.slepian)

    def _create_name(self) -> None:
        self.name = (
            f"slepian_gaussian_noise_south_america{filename_args(self.SNR, 'snr')}"
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
            self.sigma = self.extra_args[0]

    @property  # type:ignore
    def SNR(self) -> float:
        return self._SNR

    @SNR.setter
    def SNR(self, SNR: float) -> None:
        if isinstance(SNR, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            SNR = SlepianNoiseGaussianSouthAmerica._SNR
        self._SNR = SNR
