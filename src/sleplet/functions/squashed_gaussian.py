"""Contains the `SquashedGaussian` class."""
import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._string_methods
import sleplet._validation
import sleplet._vars
import sleplet.harmonic_methods
from sleplet.functions.flm import Flm


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class SquashedGaussian(Flm):
    r"""
    Creates a squashed Gaussian
    \(\exp(-\frac{1}{2}{(\frac{\theta-\overline{\theta}}{\sigma_{\theta}})}^{2})
    \sin{(\nu_{\phi}\phi)}\).
    """

    freq: float = 0.1
    r"""Sets the \(\nu_{\phi}\) value."""
    t_sigma: float = 1
    r"""Sets the \(\sigma_{\theta}\) value."""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        return sleplet.harmonic_methods._ensure_f_bandlimited(
            self._grid_fun,
            self.L,
            reality=self.reality,
            spin=self.spin,
        )

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.t_sigma, 'tsig')}"
            f"{sleplet._string_methods.filename_args(self.freq, 'freq')}"
        )

    def _set_reality(self) -> bool:
        return True

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.t_sigma, self.freq = (np.float_power(10, x) for x in self.extra_args)

    def _grid_fun(
        self,
        theta: npt.NDArray[np.float_],
        phi: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        """Function on the grid."""
        return np.exp(
            -(((theta - sleplet._vars.THETA_0) / self.t_sigma) ** 2) / 2,
        ) * np.sin(self.freq * phi)
