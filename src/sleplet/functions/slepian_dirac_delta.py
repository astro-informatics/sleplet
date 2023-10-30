"""Contains the `SlepianDiracDelta` class."""
import logging

import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import pyssht as ssht

import sleplet._string_methods
import sleplet._validation
import sleplet._vars
import sleplet.slepian_methods
from sleplet.functions.fp import Fp

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class SlepianDiracDelta(Fp):
    """Create a Dirac delta of the Slepian coefficients."""

    _alpha: float = pydantic.Field(default=0, init_var=False, repr=False)
    _beta: float = pydantic.Field(default=0, init_var=False, repr=False)

    def __post_init__(self: typing_extensions.Self) -> None:
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        self._compute_angles()
        return sleplet.slepian_methods._compute_s_p_omega_prime(
            self.L,
            self._alpha,
            self._beta,
            self.slepian,
        ).conj()

    def _create_name(self: typing_extensions.Self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"_{self.slepian.region._name_ending}"
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return False

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            msg = f"{self.__class__.__name__} does not support extra arguments"
            raise TypeError(msg)

    def _compute_angles(self: typing_extensions.Self) -> None:
        """Compute alpha/beta if not provided."""
        thetas, phis = ssht.sample_positions(
            self.L,
            Grid=True,
            Method=sleplet._vars.SAMPLING_SCHEME,
        )
        sp = ssht.inverse(
            self.slepian.eigenvectors[0],
            self.L,
            Method=sleplet._vars.SAMPLING_SCHEME,
        )
        idx = tuple(np.argwhere(sp == sp.max())[0])
        self._alpha = phis[idx]
        self._beta = thetas[idx]
        msg = (
            f"angles: (alpha, beta) = ({self._alpha/np.pi:.5f},"
            f"{self._beta/np.pi:.5f})\n"
            f"grid point: (alpha, beta) = ({self._alpha:e},{self._beta:e})"
        )
        _logger.info(msg)
