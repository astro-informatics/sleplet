from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pyssht as ssht

from pys2sleplet.flm.functions import Functions


@dataclass
class DiracDelta(Functions):
    L: int
    reality: bool = field(default=True)

    def _setup_args(self, args: Optional[List[int]]) -> None:
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

    def _create_annotations(self) -> List[Dict]:
        pass
