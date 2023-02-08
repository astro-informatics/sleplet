from dataclasses import KW_ONLY

from pydantic.dataclasses import dataclass

from sleplet.utils.mesh_methods import (
    create_mesh_region,
    extract_mesh_config,
    mesh_eigendecomposition,
    read_mesh,
)
from sleplet.utils.plotly_methods import create_camera
from sleplet.utils.validation import Validation


@dataclass(config=Validation)
class Mesh:
    name: str
    _: KW_ONLY
    number_basis_functions: int | None = None
    zoom: bool = False

    def __post_init_post_parse__(self) -> None:
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
