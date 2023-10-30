"""Contains the `SlepianNoiseAfrica` class."""
import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._string_methods
import sleplet._validation
import sleplet.functions.slepian_africa
import sleplet.noise
import sleplet.slepian.region
from sleplet.functions.fp import Fp


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class SlepianNoiseAfrica(Fp):
    """
    Create a noised Slepian region on the topographic map of the Earth of
    the Africa region.
    """

    SNR: float = -10
    """A parameter which controls the level of signal-to-noise in the noised
    data."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()
        if (
            isinstance(self.region, sleplet.slepian.region.Region)
            and self.region._name_ending != "africa"
        ):
            msg = "Slepian region selected must be 'africa'"
            raise RuntimeError(msg)

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        sa = sleplet.functions.slepian_africa.SlepianAfrica(
            self.L,
            region=self.region,
            smoothing=self.smoothing,
        )
        noise = sleplet.noise._create_slepian_noise(
            self.L,
            sa.coefficients,
            self.slepian,
            self.SNR,
        )
        sleplet.noise.compute_snr(sa.coefficients, noise, "Slepian")
        return noise

    def _create_name(self: typing_extensions.Self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.SNR, 'snr')}"
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return False

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                msg = f"The number of extra arguments should be {num_args}"
                raise ValueError(msg)
            self.SNR = self.extra_args[0]
