from abc import abstractmethod

import numpy as np
from numpy import typing as npt
from pydantic.dataclasses import dataclass

from sleplet.meshes.mesh_coefficients import MeshCoefficients
from sleplet.utils.noise import compute_snr, create_mesh_noise
from sleplet.utils.validation import Validation


@dataclass(config=Validation)
class MeshHarmonicCoefficients(MeshCoefficients):
    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _add_noise_to_signal(
        self,
    ) -> tuple[npt.NDArray[np.complex_ | np.float_] | None, float | None]:
        """
        adds Gaussian white noise to the signal
        """
        self.coefficients: npt.NDArray[np.complex_ | np.float_]
        if self.noise is not None:
            unnoised_coefficients = self.coefficients.copy()
            nlm = create_mesh_noise(self.coefficients, self.noise)
            snr = compute_snr(self.coefficients, nlm, "Harmonic")
            self.coefficients = self.coefficients + nlm
            return unnoised_coefficients, snr
        return None, None

    @abstractmethod
    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        raise NotImplementedError
