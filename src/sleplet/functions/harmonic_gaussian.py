"""Contains the `HarmonicGaussian` class."""
import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import pyssht as ssht

import sleplet._string_methods
import sleplet._validation
from sleplet.functions.flm import Flm


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class HarmonicGaussian(Flm):
    r"""
    Create a harmonic Gaussian
    \(\exp(-(\frac{{\ell}^{2}}{2\sigma_{\ell}^{2}}
    + \frac{{m}^{2}}{2\sigma_{m}^{2}}))\).
    """

    l_sigma: float = 10
    r"""Sets the \(\sigma_{\ell}\) value."""
    m_sigma: float = 10
    r"""Sets the \(\sigma_{m}\) value."""

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        flm = np.zeros(self.L**2, dtype=np.complex_)
        for ell in range(self.L):
            upsilon_l = np.exp(-((ell / self.l_sigma) ** 2) / 2)
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                flm[ind] = upsilon_l * np.exp(-((m / self.m_sigma) ** 2) / 2)
        return flm

    def _create_name(self: typing_extensions.Self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.l_sigma, 'lsig')}"
            f"{sleplet._string_methods.filename_args(self.m_sigma, 'msig')}"
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return False

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 2
            if len(self.extra_args) != num_args:
                msg = f"The number of extra arguments should be {num_args}"
                raise ValueError(msg)
            self.l_sigma, self.m_sigma = (
                np.float_power(10, x) for x in self.extra_args
            )
