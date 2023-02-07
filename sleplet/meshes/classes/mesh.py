from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from plotly.graph_objs.layout.scene import Camera

from sleplet.utils.mesh_methods import (
    create_mesh_region,
    extract_mesh_config,
    mesh_eigendecomposition,
    read_mesh,
)
from sleplet.utils.plotly_methods import create_camera


@dataclass
class Mesh:
    name: str
    number_basis_functions: Optional[int]
    zoom: bool
    _basis_functions: np.ndarray = field(init=False, repr=False)
    _camera_view: Camera = field(init=False, repr=False)
    _colourbar_pos: float = field(init=False, repr=False)
    _faces: np.ndarray = field(init=False, repr=False)
    _mesh_eigenvalues: np.ndarray = field(init=False, repr=False)
    _name: str = field(init=False, repr=False)
    _number_basis_functions: Optional[int] = field(default=None, init=False, repr=False)
    _region: np.ndarray = field(init=False, repr=False)
    _vertices: np.ndarray = field(init=False, repr=False)
    _zoom: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        mesh_config = extract_mesh_config(self.name)
        self.camera_view = create_camera(
            mesh_config.CAMERA_X,
            mesh_config.CAMERA_Y,
            mesh_config.CAMERA_Z,
            mesh_config.REGION_ZOOM if self.zoom else mesh_config.DEFAULT_ZOOM,
            x_center=mesh_config.CENTER_X if self.zoom else 0,
            y_center=mesh_config.CENTER_Y if self.zoom else 0,
            z_center=mesh_config.CENTER_Z if self.zoom else 0,
        )
        self.colourbar_pos = (
            mesh_config.REGION_COLOURBAR_POS
            if self.zoom
            else mesh_config.DEFAULT_COLOURBAR_POS
        )
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
            number_basis_functions=self.number_basis_functions,
        )
