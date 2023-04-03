"""Contains the `Slepian` class."""
import logging

import numpy as np
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass

import sleplet._validation
import sleplet.slepian_methods
from sleplet.functions.fp import Fp

_logger = logging.getLogger(__name__)


@dataclass(config=sleplet._validation.Validation, kw_only=True)
class Slepian(Fp):
    """Creates Slepian functions of the selected region."""

    rank: int = 0
    r"""Slepian eigenvalues are ordered in decreasing value. The option `rank`
    selects a given Slepian function from the spectrum (p in the papers)."""

    def __post_init_post_parse__(self) -> None:
        self._validate_rank()
        super().__post_init_post_parse__()

    def _create_name(self) -> str:
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

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        _logger.info(f"Shannon number: {self.slepian.N}")
        _logger.info(
            f"Eigenvalue {self.rank}: {self.slepian.eigenvalues[self.rank]:e}",
        )
        return sleplet.slepian_methods.slepian_forward(
            self.L,
            self.slepian,
            flm=self.slepian.eigenvectors[self.rank],
        )

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(
                    f"The number of extra arguments should be 1 or {num_args}",
                )
            self.rank = self.extra_args[0]

    def _validate_rank(self) -> None:
        """Checks the requested rank is valid."""
        if isinstance(self.extra_args, list):
            limit = self.L**2
            if self.extra_args[0] >= limit:
                raise ValueError(f"rank should be less than {limit}")

    @validator("rank")
    def _check_rank(cls, v):
        if not isinstance(v, int):
            raise TypeError("rank should be an integer")
        if v < 0:
            raise ValueError("rank cannot be negative")
        return v
