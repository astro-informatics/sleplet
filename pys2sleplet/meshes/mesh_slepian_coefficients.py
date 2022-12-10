from abc import abstractmethod
from dataclasses import dataclass, field

from sleplet.meshes.classes.mesh_slepian import MeshSlepian
from sleplet.meshes.mesh_coefficients import MeshCoefficients
from sleplet.utils.noise import compute_snr, create_slepian_mesh_noise


@dataclass  # type:ignore
class MeshSlepianCoefficients(MeshCoefficients):
    _slepian_mesh: MeshSlepian = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mesh_slepian = MeshSlepian(self.mesh)
        super().__post_init__()

    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise converted to Slepian space
        """
        if self.noise is not None:
            self.unnoised_coefficients = self.coefficients.copy()
            n_p = create_slepian_mesh_noise(
                self.mesh_slepian, self.coefficients, self.noise
            )
            self.snr = compute_snr(self.coefficients, n_p, "Slepian")
            self.coefficients += n_p

    @property
    def mesh_slepian(self) -> MeshSlepian:
        return self._slepian_mesh

    @mesh_slepian.setter
    def mesh_slepian(self, mesh_slepian: MeshSlepian) -> None:
        self._slepian_mesh = mesh_slepian

    @abstractmethod
    def _create_coefficients(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        raise NotImplementedError
