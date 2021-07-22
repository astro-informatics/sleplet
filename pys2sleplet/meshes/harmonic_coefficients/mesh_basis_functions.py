from dataclasses import dataclass, field

from pys2sleplet.meshes.mesh_harmonic_coefficients import MeshHarmonicCoefficients
from pys2sleplet.utils.harmonic_methods import mesh_forward
from pys2sleplet.utils.logger import logger


@dataclass
class MeshBasisFunctions(MeshHarmonicCoefficients):
    rank: int
    _rank: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._validate_rank()
        super().__post_init__()

    def _create_coefficients(self) -> None:
        """
        compute field on the vertices of the mesh
        """
        basis_function = self.mesh.basis_functions[self.rank]
        self.coefficients = mesh_forward(self.mesh, basis_function)
        logger.info(
            f"Mesh eigenvalue {self.rank}: "
            f"{self.mesh.mesh_eigenvalues[self.rank]:e}"
        )

    def _create_name(self) -> None:
        self.name = (
            (
                f"{self.mesh.name}_rank{self.rank}_"
                f"lam{self.mesh.mesh_eigenvalues[self.rank]:e}"
            )
            .replace(".", "-")
            .replace("+", "")
        )

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(
                    f"The number of extra arguments should be 1 or {num_args}"
                )
            self.rank = self.extra_args[0]

    def _validate_rank(self) -> None:
        """
        checks the requested rank is valid
        """
        if isinstance(self.extra_args, list):
            limit = self.mesh.mesh_eigenvalues.shape[0]
            if self.extra_args[0] > limit:
                raise ValueError(f"rank should be less than or equal to {limit}")

    @property  # type:ignore
    def rank(self) -> int:
        return self._rank

    @rank.setter
    def rank(self, rank: int) -> None:
        if isinstance(rank, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            rank = MeshBasisFunctions._rank
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        self._rank = rank
