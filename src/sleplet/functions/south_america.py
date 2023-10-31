"""Contains the `SouthAmerica` class."""
import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import pyssht as ssht

import sleplet._data.create_earth_flm
import sleplet._data.setup_pooch
import sleplet._mask_methods
import sleplet._string_methods
import sleplet._validation
import sleplet._vars
import sleplet.harmonic_methods
from sleplet.functions.flm import Flm


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class SouthAmerica(Flm):
    """Create the South America region of the topographic map of the Earth."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        return sleplet.harmonic_methods._ensure_f_bandlimited(
            self._grid_fun,
            self.L,
            reality=self.reality,
            spin=self.spin,
        )

    def _create_name(self: typing_extensions.Self) -> str:
        return sleplet._string_methods._convert_camel_case_to_snake_case(
            self.__class__.__name__,
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return True

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            msg = f"{self.__class__.__name__} does not support extra arguments"
            raise TypeError(msg)

    def _grid_fun(
        self: typing_extensions.Self,
        theta: npt.NDArray[np.float_],  # noqa: ARG002
        phi: npt.NDArray[np.float_],  # noqa: ARG002
    ) -> npt.NDArray[np.float_]:
        """Define the function on the grid."""
        earth_flm = sleplet._data.create_earth_flm.create_flm(
            self.L,
            smoothing=self.smoothing,
        )
        rot_flm = sleplet.harmonic_methods.rotate_earth_to_south_america(
            earth_flm,
            self.L,
        )
        earth_f = ssht.inverse(
            rot_flm,
            self.L,
            Method=sleplet._vars.SAMPLING_SCHEME,
            Reality=self.reality,
        )
        mask_name = f"{self.name}_L{self.L}.npy"
        mask_location = sleplet._data.setup_pooch.find_on_pooch_then_local(
            f"slepian_masks_{mask_name}",
        )
        mask = (
            sleplet._mask_methods.create_mask(self.L, mask_name)
            if mask_location is None
            else np.load(mask_location)
        )
        return np.where(mask, earth_f, 0)
