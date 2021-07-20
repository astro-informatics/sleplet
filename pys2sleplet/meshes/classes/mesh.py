from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from plotly.graph_objs.layout.scene import Camera

from pys2sleplet.utils.mesh_methods import (
    create_mesh_region,
    extract_mesh_config,
    mesh_eigendecomposition,
    read_mesh,
)
from pys2sleplet.utils.plotly_methods import create_camera


@dataclass  # type:ignore
class Mesh:
    name: str
    mesh_laplacian: bool
    number_basis_functions: Optional[int]
    _basis_functions: np.ndarray = field(init=False, repr=False)
    _camera_view: Camera = field(init=False, repr=False)
    _colourbar_pos: float = field(init=False, repr=False)
    _faces: np.ndarray = field(init=False, repr=False)
    _mesh_eigenvalues: np.ndarray = field(init=False, repr=False)
    _mesh_laplacian: bool = field(default=True, init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _number_basis_functions: Optional[int] = field(default=None, init=False, repr=False)
    _region: np.ndarray = field(init=False, repr=False)
    _vertices: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mesh_config = extract_mesh_config(self.name)
        self.camera_view = create_camera(
            mesh_config.CAMERA_X,
            mesh_config.CAMERA_Y,
            mesh_config.CAMERA_Z,
            mesh_config.ZOOM,
        )
        self.colourbar_pos = mesh_config.COLOURBAR_POS
        self.vertices, self.faces = read_mesh(mesh_config)
        self.region = create_mesh_region(mesh_config, self.vertices)
        (
            self.mesh_eigenvalues,
            self.basis_functions,
            self.number_basis_functions,
        ) = mesh_eigendecomposition(
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
    def camera_view(self) -> Camera:
        return self._camera_view

    @camera_view.setter
    def camera_view(self, camera_view: Camera) -> None:
        self._camera_view = camera_view

    @property
    def colourbar_pos(self) -> float:
        return self._colourbar_pos

    @colourbar_pos.setter
    def colourbar_pos(self, colourbar_pos: float) -> None:
        self._colourbar_pos = colourbar_pos

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

    @property  # type:ignore
    def mesh_laplacian(self) -> bool:
        return self._mesh_laplacian

    @mesh_laplacian.setter
    def mesh_laplacian(self, mesh_laplacian: bool) -> None:
        if isinstance(mesh_laplacian, property):
            # initial value not specified, use default
            # https://stackoverflow.com/a/61480946/7359333
            mesh_laplacian = Mesh._mesh_laplacian
        self._mesh_laplacian = mesh_laplacian

    @property  # type:ignore
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = name

    @property  # type:ignore
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
