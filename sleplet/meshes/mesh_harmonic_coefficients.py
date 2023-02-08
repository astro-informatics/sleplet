from abc import abstractmethod
from typing import Optional

import numpy as np
from pydantic.dataclasses import dataclass

from sleplet.meshes.mesh_coefficients import MeshCoefficients
from sleplet.utils.noise import compute_snr, create_mesh_noise
from sleplet.utils.validation import Validation


@dataclass(config=Validation)
class MeshHarmonicCoefficients(MeshCoefficients):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _add_noise_to_signal(self) -> Optional[float]:
        """
        adds Gaussian white noise to the signal
        """
        if self.noise is not None:
            self.unnoised_coefficients = self.coefficients.copy()
            nlm = create_mesh_noise(self.coefficients, self.noise)
            self.coefficients += nlm
            return compute_snr(self.coefficients, nlm, "Harmonic")
        return None

    @abstractmethod
    def _create_coefficients(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        raise NotImplementedError
