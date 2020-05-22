from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions


@dataclass
class DiracDelta(Functions):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_annotations(self) -> List[Dict]:
        pass

    def _create_flm(self, L: int) -> np.ndarray:
        flm = np.zeros((L * L), dtype=complex)
        for ell in range(L):
            ind = ssht.elm2ind(ell, m=0)
            flm[ind] = np.sqrt((2 * ell + 1) / (4 * np.pi))
        return flm

    def _create_name(self) -> str:
        name = "dirac_delta"
        return name

    def _set_reality(self) -> bool:
        return True

    def _setup_args(self, extra_args: Optional[List[int]]) -> None:
        if extra_args is not None:
            raise AttributeError(f"Does not support extra arguments")
