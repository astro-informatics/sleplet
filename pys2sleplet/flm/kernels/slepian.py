from typing import List, Optional

import numpy as np

from ..functions import Functions


class Slepian(Functions):
    # [0, 360, 0, 180, 0, 0, 0]
    def __init__(self, L: int, args: List[int] = None):
        super().__init__(L, args)

    def _setup_args(self, args: Optional[List[int]]) -> None:
        pass

    def _create_name(self) -> str:
        name = "slepian"
        return name

    def _create_flm(self, L: int) -> np.ndarray:
        pass
        # flm = self.eigenvectors[self.rank]
        # print(f"Eigenvalue {self.rank}: {self.eigenvalues[self.rank]:e}")
        # return flm
