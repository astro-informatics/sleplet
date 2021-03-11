from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from pys2sleplet.utils.mesh_methods import mesh_eigendecomposition, read_mesh


@dataclass  # type: ignore
class Mesh:
    number: Optional[int]
    extra_args: Optional[List[int]]
    _eigenvalues: np.ndarray = field(init=False, repr=False)
    _eigenvectors: np.ndarray = field(init=False, repr=False)
    _extra_args: Optional[List[int]] = field(default=None, init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _number: Optional[int] = field(default=100, init=False, repr=False)
    _triangles: np.ndarray = field(init=False, repr=False)
    _vertices: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._create_name()
        self.vertices, self.triangles = read_mesh(self.name)
        self.eigenvalues, self.eigenvectors = mesh_eigendecomposition(
            self.vertices, self.triangles, self.number
        )

    @property
    def eigenvalues(self) -> np.ndarray:
        return self._eigenvalues

    @eigenvalues.setter
    def eigenvalues(self, eigenvalues: np.ndarray) -> None:
        self._eigenvalues = eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        return self._eigenvectors

    @eigenvectors.setter
    def eigenvectors(self, eigenvectors: np.ndarray) -> None:
        self._eigenvectors = eigenvectors

    @property  # type:ignore
    def extra_args(self) -> Optional[List[int]]:
        return self._extra_args

    @extra_args.setter
    def extra_args(self, extra_args: Optional[List[int]]) -> None:
        if isinstance(extra_args, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            extra_args = Mesh._extra_args
        self._extra_args = extra_args

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property  # type: ignore
    def number(self) -> Optional[int]:
        return self._number

    @number.setter
    def number(self, number: Optional[int]) -> None:
        if isinstance(number, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            number = Mesh._number
        self._number = number

    @property
    def triangles(self) -> np.ndarray:
        return self._triangles

    @triangles.setter
    def triangles(self, triangles: np.ndarray) -> None:
        self._triangles = triangles

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @vertices.setter
    def vertices(self, vertices: np.ndarray) -> None:
        self._vertices = vertices

    @abstractmethod
    def _create_name(self) -> None:
        """
        creates the name of the mesh
        """
        raise NotImplementedError
