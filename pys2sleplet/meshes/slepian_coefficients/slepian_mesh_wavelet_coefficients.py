from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.meshes.slepian_coefficients.slepian_mesh_field import SlepianMeshField
from pys2sleplet.meshes.slepian_coefficients.slepian_mesh_wavelets import (
    SlepianMeshWavelets,
)
from pys2sleplet.utils.wavelet_methods import slepian_wavelet_forward


@dataclass
class SlepianMeshWaveletCoefficients:
    slepian_mesh_field: SlepianMeshField
    slepian_mesh_wavelets: SlepianMeshWavelets
    _slepian_mesh_field: SlepianMeshField = field(init=False, repr=False)
    _slepian_mesh_wavelets: SlepianMeshWavelets = field(init=False, repr=False)
    _wavelet_coefficients: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._create_wavelet_coefficients()

    def _create_wavelet_coefficients(self) -> None:
        """
        computes wavelet coefficients in Slepian space
        """
        self.wavelet_coefficients = slepian_wavelet_forward(
            self.slepian_mesh_field.slepian_coefficients,
            self.slepian_mesh_wavelets.wavelets,
            self.slepian_mesh_wavelets.slepian_mesh.N,
        )

    @property  # type: ignore
    def slepian_mesh_field(self) -> np.ndarray:
        return self._slepian_mesh_field

    @slepian_mesh_field.setter
    def slepian_mesh_field(self, slepian_mesh_field: np.ndarray) -> None:
        self._slepian_mesh_field = slepian_mesh_field

    @property  # type: ignore
    def slepian_mesh_wavelets(self) -> np.ndarray:
        return self._slepian_mesh_wavelets

    @slepian_mesh_wavelets.setter
    def slepian_mesh_wavelets(self, slepian_mesh_wavelets: np.ndarray) -> None:
        self._slepian_mesh_wavelets = slepian_mesh_wavelets

    @property
    def wavelet_coefficients(self) -> np.ndarray:
        return self._wavelet_coefficients

    @wavelet_coefficients.setter
    def wavelet_coefficients(self, wavelet_coefficients: np.ndarray) -> None:
        self._wavelet_coefficients = wavelet_coefficients
