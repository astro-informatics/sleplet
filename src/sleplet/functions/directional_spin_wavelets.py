"""Contains the `DirectionalSpinWavelets` class."""
import logging

import numpy as np
import pyssht as ssht
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass
from pys2let import pys2let_j_max, wavelet_tiling

import sleplet._string_methods
import sleplet._validation
from sleplet.functions.flm import Flm

_logger = logging.getLogger(__name__)


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class DirectionalSpinWavelets(Flm):
    """
    Creates directional spin scale-discretised wavelets.
    As seen in <https://doi.org/10.1016/j.acha.2016.03.009>.
    """

    B: int = 3
    r"""The wavelet parameter. Represented as \(\lambda\) in the papers."""
    j_min: int = 2
    r"""The minimum wavelet scale. Represented as \(J_{0}\) in the papers."""
    j: int | None = None
    """Option to select a given wavelet. `None` indicates the scaling function,
    whereas `0` would correspond to the selected `j_min`."""
    N: int = 2
    """Azimuthal/directional band-limit; N > 1."""
    spin: int = 0
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
            f"{sleplet._string_methods.filename_args(self.N, 'N')}"
            f"{sleplet._string_methods.wavelet_ending(self.j_min, self.j)}"
        )

    def _set_reality(self) -> bool:
        return self.j is None or self.spin == 0

    def _set_spin(self) -> int:
        return self.spin

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 5
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.spin, self.N, self.j = self.extra_args

    def _create_wavelets(self) -> npt.NDArray[np.complex_]:
        """Compute all wavelets."""
        phi_l, psi_lm = wavelet_tiling(self.B, self.L, self.N, self.j_min, self.spin)
        wavelets = np.zeros((psi_lm.shape[1] + 1, self.L**2), dtype=np.complex_)
        for ell in range(self.L):
            ind = ssht.elm2ind(ell, 0)
            wavelets[0, ind] = phi_l[ell]
        wavelets[1:] = psi_lm.T
        return wavelets

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
