"""Contains the `NoiseEarth` class."""
import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._string_methods
import sleplet._validation
import sleplet.functions.earth
import sleplet.noise
from sleplet.functions.flm import Flm


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class NoiseEarth(Flm):
    """Create a noised signal of the topographic map of the Earth."""

    SNR: float = 10
    """A parameter which controls the level of signal-to-noise in the noised
    data."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        earth = sleplet.functions.earth.Earth(self.L, smoothing=self.smoothing)
        noise = sleplet.noise._create_noise(self.L, earth.coefficients, self.SNR)
        sleplet.noise.compute_snr(earth.coefficients, noise, "Harmonic")
        return noise

    def _create_name(self: typing_extensions.Self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.SNR, 'snr')}"
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return True

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                msg = f"The number of extra arguments should be {num_args}"
                raise ValueError(msg)
            self.SNR = self.extra_args[0]
