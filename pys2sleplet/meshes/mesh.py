from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pys2sleplet.utils.mesh_methods import mesh_eigendecomposition, read_mesh


@dataclass  # type: ignore
class Mesh:
    number: int
    extra_args: Optional[list[int]]
    _eigenvalues: np.ndarray = field(init=False, repr=False)
    _eigenvectors: np.ndarray = field(init=False, repr=False)
    _extra_args: Optional[list[int]] = field(default=None, init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _number: int = field(default=0, init=False, repr=False)
    _region: np.ndarray = field(init=False, repr=False)
    _faces: np.ndarray = field(init=False, repr=False)
    _vertices: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._create_name()
        self._setup_args()
        self._solve_eigenproblem()
        self._setup_region()

    def _setup_args(self) -> None:
        if isinstance(self.extra_args, list):
            num_args = 1
            if len(self.extra_args) != num_args:
                raise ValueError(f"The number of extra arguments should be {num_args}")
            self.number = self.extra_args[0]

    def _solve_eigenproblem(self) -> None:
        """
        reads in the mesh and computes the eigendecomposition
        """
        self.vertices, self.faces = read_mesh(self.name)
        self.eigenvalues, self.eigenvectors = mesh_eigendecomposition(
            self.vertices, self.faces, self.number + 1
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
    def extra_args(self) -> Optional[list[int]]:
        return self._extra_args

    @extra_args.setter
    def extra_args(self, extra_args: Optional[list[int]]) -> None:
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
    def number(self) -> int:
        return self._number

    @number.setter
    def number(self, number: int) -> None:
        if isinstance(number, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            number = Mesh._number
        self._number = number

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

    @abstractmethod
    def _create_name(self) -> None:
        """
        creates the name of the mesh
        """
        raise NotImplementedError

    @abstractmethod
    def _setup_region(self) -> None:
        """
        creates a Slepian region on the mesh
        """
        raise NotImplementedError
