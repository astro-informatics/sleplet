"""Contains the `MeshBasisFunctions` class."""
import logging

import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._validation
import sleplet.harmonic_methods
from sleplet.meshes.mesh_harmonic_coefficients import MeshHarmonicCoefficients

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation, kw_only=True)
class MeshBasisFunctions(MeshHarmonicCoefficients):
    """Create the eigenfunctions of the Laplacian of the mesh."""

    rank: int = 0
    """Slepian eigenvalues are ordered in decreasing value. The option `rank`
    selects a given Slepian function from the spectrum (p in the papers)."""

    def __post_init__(self: typing_extensions.Self) -> None:
        self._validate_rank()
        super().__post_init__()

    def _create_coefficients(
        self: typing_extensions.Self,
    ) -> npt.NDArray[np.complex_ | np.float_]:
        """Compute field on the vertices of the mesh."""
        msg = (
            f"Mesh eigenvalue {self.rank}: "
            f"{self.mesh.mesh_eigenvalues[self.rank]:e}",
        )
        _logger.info(msg)
        basis_function = self.mesh.basis_functions[self.rank]
        return sleplet.harmonic_methods.mesh_forward(self.mesh, basis_function)

    def _create_name(self: typing_extensions.Self) -> str:
        return (
            (
                f"{self.mesh.name}_rank{self.rank}_"
                f"lam{self.mesh.mesh_eigenvalues[self.rank]:e}"
            )
            .replace(".", "-")
            .replace("+", "")
        )

    def _setup_args(self: typing_extensions.Self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                msg = f"The number of extra arguments should be 1 or {num_args}"
                raise ValueError(msg)
            self.rank = self.extra_args[0]

    def _validate_rank(self: typing_extensions.Self) -> None:
        """Check the requested rank is valid."""
        if isinstance(self.extra_args, list):
            limit = self.mesh.mesh_eigenvalues.shape[0]
            if self.extra_args[0] > limit:
                msg = f"rank should be less than or equal to {limit}"
                raise ValueError(msg)

    @pydantic.field_validator("rank")
    def _check_rank(
        cls,  # noqa: ANN101
        v: int,
    ) -> int:
        if not isinstance(v, int):
            msg = "rank should be an integer"
            raise TypeError(msg)
        if v < 0:
            msg = "rank cannot be negative"
            raise ValueError(msg)
        return v
