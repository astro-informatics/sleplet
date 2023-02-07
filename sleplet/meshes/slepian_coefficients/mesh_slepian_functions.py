from dataclasses import dataclass, field

from sleplet.meshes.mesh_slepian_coefficients import MeshSlepianCoefficients
from sleplet.utils.logger import logger
from sleplet.utils.slepian_methods import slepian_mesh_forward


@dataclass
class MeshSlepianFunctions(MeshSlepianCoefficients):
    rank: int
    _rank: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()

    def _create_coefficients(self) -> None:
        """
        compute field on the vertices of the mesh
        """
        s_p_i = self.mesh_slepian.slepian_functions[self.rank]
        self.coefficients = slepian_mesh_forward(
            self.mesh_slepian,
            u_i=s_p_i,
        )
        logger.info(
            f"Slepian eigenvalue {self.rank}: "
            f"{self.mesh_slepian.slepian_eigenvalues[self.rank]:e}"
        )

    def _create_name(self) -> None:
        self.name = (
            (
                f"slepian_{self.mesh.name}_rank{self.rank}_"
                f"lam{self.mesh_slepian.slepian_eigenvalues[self.rank]:e}"
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

    @rank.setter
    def rank(self, rank: int) -> None:
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        self._rank = rank
