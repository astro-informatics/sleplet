from dataclasses import dataclass

import numpy as np

from pys2sleplet.functions.f_p import F_P


@dataclass
class SlepianIdentity(F_P):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        self.annotations = self.slepian.annotations

    def _create_coefficients(self) -> None:
        self.coefficients = np.ones(self.L ** 2, dtype=np.complex128)

    def _create_name(self) -> None:
        self.name = f"slepian_identity_{self.slepian.region.name_ending}"

    def _set_reality(self) -> None:
        self.reality = False

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
