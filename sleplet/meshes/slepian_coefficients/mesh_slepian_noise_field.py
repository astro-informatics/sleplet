import numpy as np
from pydantic.dataclasses import dataclass

from sleplet.meshes.mesh_slepian_coefficients import MeshSlepianCoefficients
from sleplet.meshes.slepian_coefficients.mesh_slepian_field import MeshSlepianField
from sleplet.utils.noise import compute_snr, create_slepian_mesh_noise
from sleplet.utils.string_methods import filename_args
from sleplet.utils.validation import Validation


@dataclass(config=Validation, kw_only=True)
class MeshSlepianNoiseField(MeshSlepianCoefficients):
    SNR: float = -5

    def __post_init_post_parse__(self) -> None:
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> np.ndarray:
        smf = MeshSlepianField(self.mesh, region=True)
        noise = create_slepian_mesh_noise(self.mesh_slepian, smf.coefficients, self.SNR)
        compute_snr(smf.coefficients, noise, "Slepian")
        return noise

    def _create_name(self) -> str:
        return f"slepian_{self.mesh.name}_noise_field{filename_args(self.SNR, 'snr')}"

    def _set_reality(self) -> bool:
        return False

    def _set_spin(self) -> int:
        return 0

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.SNR = self.extra_args[0]
