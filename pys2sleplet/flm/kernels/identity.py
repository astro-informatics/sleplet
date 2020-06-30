from dataclasses import dataclass

import numpy as np

from pys2sleplet.flm.functions import Functions


@dataclass
class Identity(Functions):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> None:
        pass

    def _create_flm(self) -> None:
        self.multipole = np.ones(self.L * self.L, dtype=complex)

    def _create_name(self) -> None:
        self.name = "identity"

    def _set_reality(self) -> None:
        self.reality = True

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            raise AttributeError(
                f"{self.__class__.__name__} does not support extra arguments"
            )
