"""Contains the `SlepianDiracDelta` class."""
import logging

import numpy as np
import numpy.typing as npt
import pydantic.v1 as pydantic

import pyssht as ssht

import sleplet._string_methods
import sleplet._validation
import sleplet._vars
import sleplet.slepian_methods
from sleplet.functions.fp import Fp

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.Validation)
class SlepianDiracDelta(Fp):
    """Creates a Dirac delta of the Slepian coefficients."""

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        self._compute_angles()
        return sleplet.slepian_methods._compute_s_p_omega_prime(
            self.L,
            self._alpha,
            self._beta,
            self.slepian,
        ).conj()

    def _create_name(self) -> str:
        return (
            f"{sleplet._string_methods._convert_camel_case_to_snake_case(self.__class__.__name__)}"
            f"_{self.slepian.region.name_ending}"
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments",
            )

    def _compute_angles(self) -> None:
        """Computes alpha/beta if not provided."""
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
        _logger.info(
            f"angles: (alpha, beta) = ({self._alpha/np.pi:.5f},{self._beta/np.pi:.5f})",
        )
        _logger.info(
            f"grid point: (alpha, beta) = ({self._alpha:e},{self._beta:e})",
        )
