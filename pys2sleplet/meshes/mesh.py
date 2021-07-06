from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pys2sleplet.utils.mesh_methods import (
    create_mesh_region,
    mesh_eigendecomposition,
    read_mesh,
)
from pys2sleplet.utils.vars import MESH_LAPLACIAN_DEFAULT


@dataclass  # type: ignore
class Mesh:
    name: str
    mesh_laplacian: bool
    number_basis_functions: Optional[int]
    _basis_functions: np.ndarray = field(init=False, repr=False)
    _mesh_eigenvalues: np.ndarray = field(init=False, repr=False)
    _mesh_laplacian: bool = field(
        default=MESH_LAPLACIAN_DEFAULT, init=False, repr=False
    )
    _name: str = field(init=False, repr=False)
    _number_basis_functions: Optional[int] = field(default=None, init=False, repr=False)
    _region: np.ndarray = field(init=False, repr=False)
    _faces: np.ndarray = field(init=False, repr=False)
    _vertices: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertices, self.faces = read_mesh(self.name)
        self.region = create_mesh_region(self.name, self.vertices)
        self.mesh_eigenvalues, self.basis_functions = mesh_eigendecomposition(
            self.name,
            self.vertices,
            self.faces,
            mesh_laplacian=self.mesh_laplacian,
            number_basis_functions=self.number_basis_functions,
        )

    @property
    def basis_functions(self) -> np.ndarray:
        return self._basis_functions

    @basis_functions.setter
    def basis_functions(self, basis_functions: np.ndarray) -> None:
        self._basis_functions = basis_functions

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    @faces.setter
    def faces(self, faces: np.ndarray) -> None:
        self._faces = faces

    @property
    def mesh_eigenvalues(self) -> np.ndarray:
        return self._mesh_eigenvalues

    @mesh_eigenvalues.setter
    def mesh_eigenvalues(self, mesh_eigenvalues: np.ndarray) -> None:
        self._mesh_eigenvalues = mesh_eigenvalues

    @property  # type: ignore
    def mesh_laplacian(self) -> bool:
        return self._mesh_laplacian

    @mesh_laplacian.setter
    def mesh_laplacian(self, mesh_laplacian: bool) -> None:
        if isinstance(mesh_laplacian, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            mesh_laplacian = Mesh._mesh_laplacian
        self._mesh_laplacian = mesh_laplacian

    @property  # type: ignore
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property  # type: ignore
    def number_basis_functions(self) -> Optional[int]:
        return self._number_basis_functions

    @number_basis_functions.setter
    def number_basis_functions(self, number_basis_functions: Optional[int]) -> None:
        if isinstance(number_basis_functions, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            number_basis_functions = Mesh._number_basis_functions
        self._number_basis_functions = number_basis_functions

    @property
    def region(self) -> np.ndarray:
        return self._region

    @region.setter
    def region(self, region: np.ndarray) -> None:
        self._region = region

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, vertices: np.ndarray) -> None:
        self._vertices = vertices
