from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from pys2sleplet.flm.functions import Functions


@dataclass
class Identity(Functions):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> List[Dict]:
        pass

    def _create_flm(self, L: int) -> np.ndarray:
        flm = np.ones((L * L)) + 1j * np.zeros((L * L))
        return flm

    def _create_name(self) -> str:
        name = "identity"
        return name

    def _set_reality(self) -> bool:
        return True

    def _setup_args(self, extra_args: Optional[List[int]]) -> None:
        if extra_args is not None:
            raise AttributeError(f"Does not support extra arguments")
