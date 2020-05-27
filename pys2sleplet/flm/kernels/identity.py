from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from pys2sleplet.flm.functions import Functions


@dataclass
class Identity(Functions):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> List[Dict]:
        pass

    def _create_flm(self) -> np.ndarray:
        flm = np.ones((self.L * self.L), dtype=complex)
        return flm

    def _create_name(self) -> str:
        name = "identity"
        return name

    def _set_reality(self) -> bool:
        return True

    def _setup_args(self) -> None:
        if self.extra_args is not None:
            raise AttributeError(f"Does not support extra arguments")
