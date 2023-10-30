import dataclasses
import logging

import numpy as np
import numpy.typing as npt
import pydantic
import typing_extensions

import sleplet._integration_methods
import sleplet._validation
import sleplet.harmonic_methods
from sleplet.meshes.mesh_slepian import MeshSlepian

_logger = logging.getLogger(__name__)


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class MeshSlepianDecomposition:
    mesh_slepian: MeshSlepian
    _: dataclasses.KW_ONLY
    mask: bool = False
    u_i: npt.NDArray[np.complex_ | np.float_] | None = None
    u: npt.NDArray[np.complex_ | np.float_] | None = None
    _method: str = pydantic.Field(default="", init_var=False, repr=False)

    def __post_init__(self: typing_extensions.Self) -> None:
        self._detect_method()

    def decompose(self: typing_extensions.Self, rank: int) -> float:
        """Decompose the signal into its Slepian coefficients via the given method."""
        self._validate_rank(rank)

        match self._method:
            case "harmonic_sum":
                return self._harmonic_sum(rank)
            case "integrate_mesh":
                return self._integrate_mesh(rank)
            case "integrate_region":
                return self._integrate_region(rank)
            case _:
                msg = f"'{self._method}' is not a valid method"
                raise ValueError(msg)

    def decompose_all(
        self: typing_extensions.Self,
        n_coefficients: int,
    ) -> npt.NDArray[np.float_]:
        """Decompose all ranks of the Slepian coefficients."""
        coefficients = np.zeros(n_coefficients)
        for rank in range(n_coefficients):
            coefficients[rank] = self.decompose(rank)
        return coefficients

    def _integrate_region(self: typing_extensions.Self, rank: int) -> float:
        r"""
        F_{p} =
        \frac{1}{\lambda_{p}}
        \int\limits_{R} \dd{x}
        f(x) \overline{S_{p}(x)}.
        """
        s_p = sleplet.harmonic_methods.mesh_inverse(
            self.mesh_slepian.mesh,
            self.mesh_slepian.slepian_functions[rank],
        )
        integration = sleplet._integration_methods.integrate_region_mesh(
            self.mesh_slepian.mesh.mesh_region,
            self.mesh_slepian.mesh.vertices,
            self.mesh_slepian.mesh.faces,
            self.u,
            s_p,
        )
        return integration / self.mesh_slepian.slepian_eigenvalues[rank]

    def _integrate_mesh(self: typing_extensions.Self, rank: int) -> float:
        r"""
        F_{p} =
        \int\limits_{x} \dd{x}
        f(x) \overline{S_{p}(x)}.
        """
        s_p = sleplet.harmonic_methods.mesh_inverse(
            self.mesh_slepian.mesh,
            self.mesh_slepian.slepian_functions[rank],
        )
        return sleplet._integration_methods.integrate_whole_mesh(
            self.mesh_slepian.mesh.vertices,
            self.mesh_slepian.mesh.faces,
            self.u,
            s_p,
        )

    def _harmonic_sum(self: typing_extensions.Self, rank: int) -> float:
        r"""
        F_{p} =
        \sum\limits_{i=0}^{K}
        f_{i} (S_{p})_{i}^{*}.
        """
        return (self.u_i * self.mesh_slepian.slepian_functions[rank]).sum()

    def _detect_method(self: typing_extensions.Self) -> None:
        """Detect what method is used to perform the decomposition."""
        if self.u_i is not None:
            _logger.info("harmonic sum method selected")
            self._method = "harmonic_sum"
        elif self.u is not None:
            if self.mask:
                _logger.info("integrating a region on the mesh method selected")
                self._method = "integrate_region"
            else:
                _logger.info("integrating the whole mesh method selected")
                self._method = "integrate_mesh"
        else:
            msg = (
                "need to pass one off harmonic coefficients, real pixels "
                "or real pixels with a mask"
            )
            raise RuntimeError(msg)

    def _validate_rank(self: typing_extensions.Self, rank: int) -> None:
        """Check the requested rank is valid."""
        if not isinstance(rank, int):
            msg = "rank should be an integer"
            raise TypeError(msg)
        if rank < 0:
            msg = "rank cannot be negative"
            raise ValueError(msg)
        if rank >= self.mesh_slepian.mesh.number_basis_functions:
            msg = (
                "rank should be less than "
                f"{self.mesh_slepian.mesh.number_basis_functions}"
            )
            raise ValueError(msg)
