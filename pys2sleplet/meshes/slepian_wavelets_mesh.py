from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.meshes.slepian_mesh import SlepianMesh
from pys2sleplet.utils.wavelet_methods import create_kappas


@dataclass
class SlepianWaveletsMesh:
    slepian_mesh: SlepianMesh
    B: int
    j_min: int
    _B: int = field(default=3, init=False, repr=False)
    _j_min: int = field(default=2, init=False, repr=False)
    _slepian_mesh: SlepianMesh = field(init=False, repr=False)
    _wavelets: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._create_wavelets()

    def _create_wavelets(self) -> None:
        """
        creates the Slepian wavelets of the mesh
        """
        self.wavelets = create_kappas(
            self.slepian_mesh.slepian_functions.shape[0], self.B, self.j_min
        )

    @property  # type:ignore
    def B(self) -> int:
        return self._B

    @B.setter
    def B(self, B: int) -> None:
        if isinstance(B, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            B = SlepianWaveletsMesh._B
        self._B = B

    @property  # type:ignore
    def j_min(self) -> int:
        return self._j_min

    @j_min.setter
    def j_min(self, j_min: int) -> None:
        if isinstance(j_min, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            j_min = SlepianWaveletsMesh._j_min
        self._j_min = j_min

    @property  # type: ignore
    def slepian_mesh(self) -> SlepianMesh:
        return self._slepian_mesh

    @slepian_mesh.setter
    def slepian_mesh(self, slepian_mesh: SlepianMesh) -> None:
        self._slepian_mesh = slepian_mesh

    @property
    def wavelets(self) -> np.ndarray:
        return self._wavelets

    @wavelets.setter
    def wavelets(self, wavelets: np.ndarray) -> None:
        self._wavelets = wavelets
