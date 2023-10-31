"""Contains the `Slepian` class."""
import logging

import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._validation
import sleplet.slepian_methods
from sleplet.functions.fp import Fp

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class Slepian(Fp):
    """Create Slepian functions of the selected region."""

    rank: int = 0
    r"""Slepian eigenvalues are ordered in decreasing value. The option `rank`
    selects a given Slepian function from the spectrum (p in the papers)."""

    def __post_init__(self: typing_extensions.Self) -> None:
        self._validate_rank()
        super().__post_init__()

    def _create_name(self: typing_extensions.Self) -> str:
        order = (
            f"_m{self.slepian.order[self.rank]}"
            if hasattr(self.slepian, "order")
            else ""
        )
        return (
            (
                f"{self.slepian.name}{order}_rank{self.rank}"
                f"_lam{self.slepian.eigenvalues[self.rank]:e}"
            )
            .replace(".", "-")
            .replace("+", "")
        )

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        msg = (
            f"Shannon number: {self.slepian.N}\n"
            f"Eigenvalue {self.rank}: {self.slepian.eigenvalues[self.rank]:e}"
        )
        _logger.info(msg)
        return sleplet.slepian_methods.slepian_forward(
            self.L,
            self.slepian,
            flm=self.slepian.eigenvectors[self.rank],
        )

    def _set_reality(self: typing_extensions.Self) -> bool:
        return False

    def _set_spin(self: typing_extensions.Self) -> int:
        return 0

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                msg = f"The number of extra arguments should be 1 or {num_args}"
                raise ValueError(msg)
            self.rank = self.extra_args[0]

    def _validate_rank(self: typing_extensions.Self) -> None:
        """Check the requested rank is valid."""
        if isinstance(self.extra_args, list):
            limit = self.L**2
            if self.extra_args[0] >= limit:
                msg = f"rank should be less than {limit}"
                raise ValueError(msg)

    @pydantic.field_validator("rank")
    def _check_rank(
        cls,  # noqa: ANN101
        v: int,
    ) -> int:
        if not isinstance(v, int):
            msg = "rank should be an integer"
            raise TypeError(msg)
        if v < 0:
            msg = "rank cannot be negative"
            raise ValueError(msg)
        return v
