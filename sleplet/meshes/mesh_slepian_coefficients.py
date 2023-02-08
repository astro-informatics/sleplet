from abc import abstractmethod

import numpy as np
from pydantic.dataclasses import dataclass

from sleplet.meshes.classes.mesh_slepian import MeshSlepian
from sleplet.meshes.mesh_coefficients import MeshCoefficients
from sleplet.utils.noise import compute_snr, create_slepian_mesh_noise
from sleplet.utils.validation import Validation


@dataclass(config=Validation)
class MeshSlepianCoefficients(MeshCoefficients):
    def __post_init_post_parse__(self) -> None:
        self.mesh_slepian = MeshSlepian(self.mesh)
        super().__post_init_post_parse__()

    def _add_noise_to_signal(self) -> tuple[np.ndarray | None, float | None]:
        """
        adds Gaussian white noise converted to Slepian space
        """
        if self.noise is not None:
            unnoised_coefficients = self.coefficients.copy()
            n_p = create_slepian_mesh_noise(
                self.mesh_slepian, self.coefficients, self.noise
            )
            snr = compute_snr(self.coefficients, n_p, "Slepian")
            self.coefficients += n_p
            return unnoised_coefficients, snr
        return None, None

    @abstractmethod
    def _create_coefficients(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        raise NotImplementedError
