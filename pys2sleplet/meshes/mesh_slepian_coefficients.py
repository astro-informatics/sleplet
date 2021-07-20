from abc import abstractmethod
from dataclasses import dataclass, field

from pys2sleplet.meshes.classes.slepian_mesh import SlepianMesh
from pys2sleplet.meshes.mesh_coefficients import MeshCoefficients
from pys2sleplet.utils.noise import compute_snr, create_slepian_mesh_noise


@dataclass  # type: ignore
class MeshSlepianCoefficients(MeshCoefficients):
    _slepian_mesh: SlepianMesh = field(init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _add_noise_to_signal(self) -> None:
        """
        adds Gaussian white noise converted to Slepian space
        """
        if self.noise is not None:
            n_p = create_slepian_mesh_noise(self.mesh)
            self.snr = compute_snr(self.coefficients, n_p, "Slepian")
            self.coefficients += n_p

    @property
    def slepian_mesh(self) -> SlepianMesh:
        return self._slepian_mesh

    @slepian_mesh.setter
    def slepian_mesh(self, slepian_mesh: SlepianMesh) -> None:
        self._slepian_mesh = slepian_mesh

    @abstractmethod
    def _create_coefficients(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _create_name(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def _setup_args(self) -> None:
        raise NotImplementedError
