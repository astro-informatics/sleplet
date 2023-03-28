import numpy as np
from numpy import typing as npt
from pydantic import validator
from pydantic.dataclasses import dataclass

from sleplet import logger
from sleplet.meshes.mesh_harmonic_coefficients import MeshHarmonicCoefficients
from sleplet.utils._validation import Validation
from sleplet.utils.harmonic_methods import _mesh_forward


@dataclass(config=Validation, kw_only=True)
class MeshBasisFunctions(MeshHarmonicCoefficients):
    rank: int = 0

    def __post_init_post_parse__(self) -> None:
        self._validate_rank()
        super().__post_init_post_parse__()

    def _create_coefficients(self) -> npt.NDArray[np.complex_ | np.float_]:
        """
        compute field on the vertices of the mesh
        """
        logger.info(
            f"Mesh eigenvalue {self.rank}: "
            f"{self.mesh.mesh_eigenvalues[self.rank]:e}"
        )
        basis_function = self.mesh.basis_functions[self.rank]
        return _mesh_forward(self.mesh, basis_function)

    def _create_name(self) -> str:
        return (
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

    @validator("rank")
    def check_rank(cls, v):
        if not isinstance(v, int):
            raise TypeError("rank should be an integer")
        if v < 0:
            raise ValueError("rank cannot be negative")
        return v
