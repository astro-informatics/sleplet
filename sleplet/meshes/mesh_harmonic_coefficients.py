from abc import abstractmethod

from pydantic.dataclasses import dataclass

from sleplet.meshes.mesh_coefficients import MeshCoefficients
from sleplet.utils.noise import compute_snr, create_mesh_noise


@dataclass
class MeshHarmonicCoefficients(MeshCoefficients):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise to the signal
        """
        if self.noise is not None:
            self.unnoised_coefficients = self.coefficients.copy()
            nlm = create_mesh_noise(self.coefficients, self.noise)
            self.snr = compute_snr(self.coefficients, nlm, "Harmonic")
            self.coefficients += nlm

    @abstractmethod
    def _create_coefficients(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        raise NotImplementedError
