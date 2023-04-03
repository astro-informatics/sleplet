"""Contains the `Ridgelets` class."""
import logging

import numpy as np
import pyssht as ssht
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass
from pys2let import pys2let_j_max
from scipy.special import gammaln

import sleplet._string_methods
import sleplet._validation
import sleplet.wavelet_methods
from sleplet.functions.flm import Flm

_logger = logging.getLogger(__name__)


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class Ridgelets(Flm):
    """
    Crates scale-discretised wavelets. As seen in
    <https://arxiv.org/abs/1510.01595>.
    """

    B: int = 3
    r"""The wavelet parameter. Represented as \(\lambda\) in the papers."""
    j_min: int = 2
    r"""The minimum wavelet scale. Represented as \(J_{0}\) in the papers."""
    j: int | None = None
    """Option to select a given wavelet. `None` indicates the scaling function,
    whereas would correspond to the selected `j_min`."""
    spin: int = 2
    """Spin value."""

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        _logger.info("start computing wavelets")
        self.wavelets = self._create_wavelets()
        _logger.info("finish computing wavelets")
        jth = 0 if self.j is None else self.j + 1
        return self.wavelets[jth]

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"{sleplet._string_methods.filename_args(self.B, 'B')}"
            f"{sleplet._string_methods.filename_args(self.j_min, 'jmin')}"
            f"{sleplet._string_methods.filename_args(self.spin, 'spin')}"
            f"{sleplet._string_methods.wavelet_ending(self.j_min, self.j)}"
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return self.spin

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 4
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.spin, self.j = self.extra_args

    def _create_wavelets(self) -> npt.NDArray[np.complex_]:
        """Compute all wavelets."""
        ring_lm = self._compute_ring()
        kappas = sleplet.wavelet_methods.create_kappas(self.L, self.B, self.j_min)
        wavelets = np.zeros((kappas.shape[0], self.L**2), dtype=np.complex_)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            wavelets[0, ind] = kappas[0, ell] * ring_lm[ind]
            wavelets[1:, ind] = kappas[1:, ell] * ring_lm[ind] / np.sqrt(2 * np.pi)
        return wavelets

    def _compute_ring(self) -> npt.NDArray[np.complex_]:
        """Compute ring in harmonic space."""
        ring_lm = np.zeros(self.L**2, dtype=np.complex_)
        for ell in range(abs(self.spin), self.L):
            logp2 = (
                gammaln(ell + self.spin + 1)
                - ell * np.log(2)
                - gammaln((ell + self.spin) / 2 + 1)
                - gammaln((ell - self.spin) / 2 + 1)
            )
            p0 = np.real((-1) ** ((ell + self.spin) / 2)) * np.exp(logp2)
            ind = ssht.elm2ind(ell, 0)
            ring_lm[ind] = (
                2
                * np.pi
                * np.sqrt((2 * ell + 1) / (4 * np.pi))
                * p0
                * (-1) ** self.spin
                * np.sqrt(
                    np.exp(gammaln(ell - self.spin + 1) - gammaln(ell + self.spin + 1)),
                )
            )
        return ring_lm

    @validator("j")
    def _check_j(cls, v, values):
        j_max = pys2let_j_max(values["B"], values["L"], values["j_min"])
        if v is not None and v < 0:
            raise ValueError("j should be positive")
        if v is not None and v > j_max - values["j_min"]:
            raise ValueError(
                f"j should be less than j_max - j_min: {j_max - values['j_min'] + 1}",
            )
        return v
