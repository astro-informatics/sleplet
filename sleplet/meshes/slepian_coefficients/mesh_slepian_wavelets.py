from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from pys2let import pys2let_j_max

from sleplet.meshes.mesh_slepian_coefficients import MeshSlepianCoefficients
from sleplet.utils.logger import logger
from sleplet.utils.string_methods import filename_args, wavelet_ending
from sleplet.utils.wavelet_methods import create_kappas


@dataclass
class MeshSlepianWavelets(MeshSlepianCoefficients):
    B: int
    j_min: int
    j: Optional[int]
    _B: int = field(default=3, init=False, repr=False)
    _j: Optional[int] = field(default=None, init=False, repr=False)
    _j_max: int = field(init=False, repr=False)
    _j_min: int = field(default=2, init=False, repr=False)
    _wavelets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        logger.info("start computing wavelets")
        self._create_wavelets()
        logger.info("finish computing wavelets")
        jth = 0 if self.j is None else self.j + 1
        self.coefficients = self.wavelets[jth]

    def _create_name(self) -> None:
        self.name = (
            f"slepian_wavelets_{self.mesh.name}"
            f"{filename_args(self.B, 'B')}"
            f"{filename_args(self.j_min, 'jmin')}"
            f"{wavelet_ending(self.j_min, self.j)}"
        )

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 3
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.B, self.j_min, self.j = self.extra_args

    def _create_wavelets(self) -> None:
        """
        creates the Slepian wavelets of the mesh
        """
        self.wavelets = create_kappas(
            self.mesh.mesh_eigenvalues.shape[0], self.B, self.j_min
        )

    @B.setter
    def B(self, B: int) -> None:
        self._B = B

    @j.setter
    def j(self, j: Optional[int]) -> None:
        self.j_max = pys2let_j_max(
            self.B, self.mesh.mesh_eigenvalues.shape[0], self.j_min
        )
        if j is not None and j < 0:
            raise ValueError("j should be positive")
        if j is not None and j > self.j_max - self.j_min:
            raise ValueError(
                f"j should be less than j_max - j_min: {self.j_max - self.j_min + 1}"
            )
        self._j = j

    @j_max.setter
    def j_max(self, j_max: int) -> None:
        self._j_max = j_max

    @j_min.setter
    def j_min(self, j_min: int) -> None:
        self._j_min = j_min

    @wavelets.setter
    def wavelets(self, wavelets: np.ndarray) -> None:
        self._wavelets = wavelets
