"""Contains the `HarmonicGaussian` class."""
import numpy as np
import pyssht as ssht
from numpy import typing as npt
from pydantic.dataclasses import dataclass

import sleplet._string_methods
import sleplet._validation
from sleplet.functions.flm import Flm


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class HarmonicGaussian(Flm):
    r"""
    Creates a harmonic Gaussian
    \(\exp(-(\frac{{\ell}^{2}}{2\sigma_{\ell}^{2}}
    + \frac{{m}^{2}}{2\sigma_{m}^{2}}))\).
    """

    l_sigma: float = 10
    r"""Sets the \(\sigma_{\ell}\) value."""
    m_sigma: float = 10
    r"""Sets the \(\sigma_{m}\) value."""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        flm = np.zeros(self.L**2, dtype=np.complex_)
        for ell in range(self.L):
            upsilon_l = np.exp(-((ell / self.l_sigma) ** 2) / 2)
            for m in range(-ell, ell + 1):
                ind = ssht.elm2ind(ell, m)
                flm[ind] = upsilon_l * np.exp(-((m / self.m_sigma) ** 2) / 2)
        return flm

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.l_sigma, 'lsig')}"
            f"{sleplet._string_methods.filename_args(self.m_sigma, 'msig')}"
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 2
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.l_sigma, self.m_sigma = (
                np.float_power(10, x) for x in self.extra_args
            )
