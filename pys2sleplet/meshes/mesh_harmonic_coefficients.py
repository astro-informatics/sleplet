from abc import abstractmethod
from dataclasses import dataclass

from pys2sleplet.meshes.mesh_coefficients import MeshCoefficients
from pys2sleplet.utils.noise import compute_snr, create_mesh_noise


@dataclass  # type:ignore
class MeshHarmonicCoefficients(MeshCoefficients):
    def __post_init__(self) -> None:
        super().__post_init__()

    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise to the signal
        """
        if self.noise is not None:
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
