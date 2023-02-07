from pydantic.dataclasses import dataclass

from sleplet.meshes.harmonic_coefficients.mesh_field import MeshField
from sleplet.meshes.mesh_harmonic_coefficients import MeshHarmonicCoefficients
from sleplet.utils.noise import compute_snr, create_mesh_noise
from sleplet.utils.string_methods import filename_args
from sleplet.utils.validation import Validation


@dataclass(config=Validation, kw_only=True)
class MeshNoiseField(MeshHarmonicCoefficients):
    SNR: int = 10

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        mf = MeshField(self.mesh)
        noise = create_mesh_noise(mf.coefficients, self.SNR)
        compute_snr(mf.coefficients, noise, "Harmonic")
        self.coefficients = noise

    def _create_name(self) -> None:
        self.name = f"{self.mesh.name}_noise_field{filename_args(self.SNR, 'snr')}"

    def _set_reality(self) -> None:
        self.reality = True

    def _set_spin(self) -> None:
        self.spin = 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.SNR = self.extra_args[0]
