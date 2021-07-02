from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.meshes.slepian_mesh_field import SlepianMeshField
from pys2sleplet.meshes.slepian_wavelets_mesh import SlepianWaveletsMesh
from pys2sleplet.utils.wavelet_methods import slepian_wavelet_forward


@dataclass
class SlepianWaveletCoefficientsMesh:
    slepian_mesh_field: SlepianMeshField
    slepian_wavelet_mesh: SlepianWaveletsMesh
    _slepian_mesh_field: SlepianMeshField = field(init=False, repr=False)
    _slepian_wavelet_mesh: SlepianWaveletsMesh = field(init=False, repr=False)
    _wavelet_coefficients: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._create_wavelet_coefficients()

    def _create_wavelet_coefficients(self) -> None:
        """
        computes wavelet coefficients in Slepian space
        """
        self.wavelet_coefficients = slepian_wavelet_forward(
            self.slepian_mesh_field.slepian_coefficients,
            self.slepian_wavelet_mesh.wavelets,
            self.slepian_wavelet_mesh.slepian_mesh.N,
        )

    @property  # type: ignore
    def slepian_mesh_field(self) -> np.ndarray:
        return self._slepian_mesh_field

    @slepian_mesh_field.setter
    def slepian_mesh_field(self, slepian_mesh_field: np.ndarray) -> None:
        self._slepian_mesh_field = slepian_mesh_field

    @property  # type: ignore
    def slepian_wavelet_mesh(self) -> np.ndarray:
        return self._slepian_wavelet_mesh

    @slepian_wavelet_mesh.setter
    def slepian_wavelet_mesh(self, slepian_wavelet_mesh: np.ndarray) -> None:
        self._slepian_wavelet_mesh = slepian_wavelet_mesh

    @property
    def wavelet_coefficients(self) -> np.ndarray:
        return self._wavelet_coefficients

    @wavelet_coefficients.setter
    def wavelet_coefficients(self, wavelet_coefficients: np.ndarray) -> None:
        self._wavelet_coefficients = wavelet_coefficients
