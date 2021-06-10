from dataclasses import dataclass, field

import numpy as np

from pys2sleplet.utils.mesh_methods import (
    create_mesh_region,
    mesh_eigendecomposition,
    read_mesh,
)
from pys2sleplet.utils.vars import LAPLACIAN_DEFAULT


@dataclass  # type: ignore
class Mesh:
    name: str
    laplacian_type: str
    _basis_functions: np.ndarray = field(init=False, repr=False)
    _laplacian_type: str = field(default=LAPLACIAN_DEFAULT, init=False, repr=False)
    _mesh_eigenvalues: np.ndarray = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _region: np.ndarray = field(init=False, repr=False)
    _faces: np.ndarray = field(init=False, repr=False)
    _vertices: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.vertices, self.faces = read_mesh(self.name)
        self.region = create_mesh_region(self.name, self.vertices)
        self.mesh_eigenvalues, self.basis_functions = mesh_eigendecomposition(
            self.name, self.vertices, self.faces, laplacian_type=self.laplacian_type
        )

    @property
    def basis_functions(self) -> np.ndarray:
        return self._basis_functions

    @basis_functions.setter
    def basis_functions(self, basis_functions: np.ndarray) -> None:
        self._basis_functions = basis_functions

    @property  # type: ignore
    def laplacian_type(self) -> str:
        return self._laplacian_type

    @laplacian_type.setter
    def laplacian_type(self, laplacian_type: str) -> None:
        self._laplacian_type = laplacian_type

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
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

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
