from abc import abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np

from pys2sleplet.flm.functions.slepian import Slepian


class SlepianFunctions(Slepian):
    def __init__(self, L: int) -> None:
        self.L = L
        self.matrix_filename = Path(f"D_L-{L}_")
        self.eigenvalues, self.eigenvectors = self.eigenproblem()

    @abstractmethod
    def eigenproblem(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def annotations(self) -> List[dict]:
        raise NotImplementedError
