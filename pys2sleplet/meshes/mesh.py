from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.utils.mesh_methods import (
    create_mesh_region,
    mesh_eigendecomposition,
    read_mesh,
)


@dataclass  # type: ignore
class Mesh:
    name: str
    num_basis_fun: int
    _basis_functions: np.ndarray = field(init=False, repr=False)
    _mesh_eigenvalues: np.ndarray = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _num_basis_fun: int = field(default=0, init=False, repr=False)
    _region: np.ndarray = field(init=False, repr=False)
    _faces: np.ndarray = field(init=False, repr=False)
    _vertices: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertices, self.faces = read_mesh(self.name)
        self.region = create_mesh_region(self.name, self.vertices)
        self.eigenvalues, self.eigenvectors = mesh_eigendecomposition(
            self.vertices, self.faces, self.num_basis_fun
        )

    @property
    def mesh_eigenvalues(self) -> np.ndarray:
        return self._mesh_eigenvalues

    @mesh_eigenvalues.setter
    def mesh_eigenvalues(self, mesh_eigenvalues: np.ndarray) -> None:
        self._mesh_eigenvalues = mesh_eigenvalues

    @property
    def basis_functions(self) -> np.ndarray:
        return self.basis_functions

    @basis_functions.setter
    def basis_functions(self, basis_functions: np.ndarray) -> None:
        self.basis_functions = basis_functions

    @property  # type: ignore
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property  # type: ignore
    def num_basis_fun(self) -> int:
        return self._num_basis_fun

    @num_basis_fun.setter
    def num_basis_fun(self, num_basis_fun: int) -> None:
        self._num_basis_fun = num_basis_fun

    @property
    def region(self) -> np.ndarray:
        return self._region

    @region.setter
    def region(self, region: np.ndarray) -> None:
        self._region = region

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    @faces.setter
    def faces(self, faces: np.ndarray) -> None:
        self._faces = faces

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, vertices: np.ndarray) -> None:
        self._vertices = vertices
