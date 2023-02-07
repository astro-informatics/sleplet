from pydantic import validator
from pydantic.dataclasses import dataclass

from sleplet.functions.f_p import F_P
from sleplet.utils.logger import logger
from sleplet.utils.slepian_methods import slepian_forward
from sleplet.utils.validation import Validation


@dataclass(config=Validation, kw_only=True)
class Slepian(F_P):
    rank: int = 0

    def __post_init__(self) -> None:
        self._validate_rank()
        super().__post_init__()

    def _create_name(self) -> None:
        order = (
            f"_m{self.slepian.order[self.rank]}"
            if hasattr(self.slepian, "order")
            else ""
        )
        self.name = (
            (
                f"{self.slepian.name}{order}_rank{self.rank}"
                f"_lam{self.slepian.eigenvalues[self.rank]:e}"
            )
            .replace(".", "-")
            .replace("+", "")
        )

    def _create_coefficients(self) -> None:
        self.coefficients = slepian_forward(
            self.L, self.slepian, flm=self.slepian.eigenvectors[self.rank]
        )
        logger.info(f"Shannon number: {self.slepian.N}")
        logger.info(f"Eigenvalue {self.rank}: {self.slepian.eigenvalues[self.rank]:e}")

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(
                    f"The number of extra arguments should be 1 or {num_args}"
                )
            self.rank = self.extra_args[0]

    def _validate_rank(self) -> None:
        """
        checks the requested rank is valid
        """
        if isinstance(self.extra_args, list):
            limit = self.L**2
            if self.extra_args[0] >= limit:
                raise ValueError(f"rank should be less than {limit}")

    @validator("rank")
    def check_rank(cls, rank: int) -> int:
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        return rank
