from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pys2sleplet.meshes.classes.mesh import Mesh
from pys2sleplet.utils.integration_methods import (
    integrate_region_mesh,
    integrate_whole_mesh,
)
from pys2sleplet.utils.logger import logger
from pys2sleplet.utils.mesh_methods import mesh_inverse


@dataclass
class SlepianMeshDecomposition:
    mesh: Mesh
    slepian_eigenvalues: np.ndarray
    slepian_functions: np.ndarray
    shannon: int
    u: Optional[np.ndarray]
    u_i: Optional[np.ndarray]
    mask: bool
    _mask: bool = field(default=False, init=False, repr=False)
    _mesh: Mesh = field(init=False, repr=False)
    _shannon: int = field(init=False, repr=False)
    _slepian_eigenvalues: np.ndarray = field(init=False, repr=False)
    _slepian_functions: np.ndarray = field(init=False, repr=False)
    _u: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _u_i: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """ """
        self._detect_method()

    def decompose(self, rank: int) -> float:
        """
        decompose the signal into its Slepian coefficients via the given method
        """
        self._validate_rank(rank)

        if self.method == "harmonic_sum":
            return self._harmonic_sum(rank)
        elif self.method == "integrate_mesh":
            return self._integrate_mesh(rank)
        elif self.method == "integrate_region":
            return self._integrate_region(rank)
        else:
            raise ValueError(f"'{self.method}' is not a valid method")

    def decompose_all(self) -> np.ndarray:
        """
        decompose all ranks of the Slepian coefficients
        """
        coefficients = np.zeros(self.shannon)
        for rank in range(self.shannon):
            coefficients[rank] = self.decompose(rank)
        return coefficients

    def _integrate_region(self, rank: int) -> float:
        r"""
        f_{p} =
        \frac{1}{\lambda_{p}}
        \int\limits_{R} \dd{x}
        f(x) \overline{S_{p}(x)}
        """
        s_p = mesh_inverse(
            self.mesh.basis_functions,
            self.slepian_functions[rank],
        )
        integration = integrate_region_mesh(self.mesh.region, self.u, s_p)
        return integration / self.slepian_eigenvalues[rank]

    def _integrate_mesh(self, rank: int) -> float:
        r"""
        f_{p} =
        \int\limits_{x} \dd{x}
        f(x) \overline{S_{p}(x)}
        """
        s_p = mesh_inverse(
            self.mesh.basis_functions,
            self.slepian_functions[rank],
        )
        return integrate_whole_mesh(self.u, s_p)

    def _harmonic_sum(self, rank: int) -> float:
        r"""
        f_{p} =
        \sum\limits_{i=0}^{K}
        f_{i} (S_{p})_{i}^{*}
        """
        return (self.u_i * self.slepian_functions[rank]).sum()

    def _detect_method(self) -> None:
        """
        detects what method is used to perform the decomposition
        """
        if isinstance(self.u_i, np.ndarray):
            logger.info("harmonic sum method selected")
            self.method = "harmonic_sum"
        elif isinstance(self.u, np.ndarray) and not self.mask:
            logger.info("integrating the whole mesh method selected")
            self.method = "integrate_mesh"
        elif isinstance(self.u, np.ndarray):
            logger.info("integrating a region on the mesh method selected")
            self.method = "integrate_region"
        else:
            raise RuntimeError(
                "need to pass one off harmonic coefficients, real pixels "
                "or real pixels with a mask"
            )

    def _validate_rank(self, rank: int) -> None:
        """
        checks the requested rank is valid
        """
        if not isinstance(rank, int):
            raise TypeError("rank should be an integer")
        if rank < 0:
            raise ValueError("rank cannot be negative")
        if rank >= self.shannon:
            raise ValueError(f"rank should be less than {self.shannon}")

    @property  # type: ignore
    def mask(self) -> bool:
        return self._mask

    @mask.setter
    def mask(self, mask: bool) -> None:
        if isinstance(mask, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            mask = SlepianMeshDecomposition._mask
        self._mask = mask

    @property  # type:ignore
    def mesh(self) -> Mesh:
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        self._mesh = mesh

    @property  # type:ignore
    def shannon(self) -> int:
        return self._shannon

    @shannon.setter
    def shannon(self, shannon: int) -> None:
        self._shannon = shannon

    @property  # type:ignore
    def slepian_eigenvalues(self) -> np.ndarray:
        return self._slepian_eigenvalues

    @slepian_eigenvalues.setter
    def slepian_eigenvalues(self, slepian_eigenvalues: np.ndarray) -> None:
        self._slepian_eigenvalues = slepian_eigenvalues

    @property  # type:ignore
    def slepian_functions(self) -> np.ndarray:
        return self._slepian_functions

    @slepian_functions.setter
    def slepian_functions(self, slepian_functions: np.ndarray) -> None:
        self._slepian_functions = slepian_functions

    @property  # type:ignore
    def mesh(self) -> Mesh:
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        self._mesh = mesh

    @property  # type:ignore
    def u(self) -> Optional[np.ndarray]:
        return self._u

    @u.setter
    def u(self, u: Optional[np.ndarray]) -> None:
        if isinstance(u, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            u = SlepianMeshDecomposition._u
        self._u = u

    @property  # type:ignore
    def u_i(self) -> Optional[np.ndarray]:
        return self._u_i

    @u_i.setter
    def u_i(self, u_i: Optional[np.ndarray]) -> None:
        if isinstance(u_i, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            u_i = SlepianMeshDecomposition._u_i
        self._u_i = u_i
