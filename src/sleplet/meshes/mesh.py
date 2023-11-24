"""Contains the `Mesh` class."""
import dataclasses

import numpy as np
import numpy.typing as npt
import plotly.graph_objs as go
import pydantic
import typing_extensions

import sleplet._mesh_methods
import sleplet._plotly_methods
import sleplet._validation


@pydantic.dataclasses.dataclass(config=sleplet._validation.validation)
class Mesh:
    """Create a mesh object."""

    name: str
    """The name of the mesh. Current options `bird`, `cheetah`, `cube`, `dragon`
    `homer` and `teapot`."""
    _: dataclasses.KW_ONLY
    number_basis_functions: int | None = None
    """The number of basis functions to compute from the Laplacian of the given
    mesh. Defaults to one quarter of the number of vertices o the given mesh."""
    zoom: bool = False
    """Whether to zoom in on the pre-selected region of the mesh in the
    plots."""
    _camera_view: go.layout.scene.Camera | None = pydantic.Field(
        default=go.layout.scene.Camera(),
        init_var=False,
        repr=False,
    )
    _colourbar_pos: float = pydantic.Field(default=0, init_var=False, repr=False)
    basis_functions: npt.NDArray[np.float_] = pydantic.Field(
        default_factory=lambda: np.empty(0),
        init_var=False,
        repr=False,
    )
    faces: npt.NDArray[np.float_] = pydantic.Field(
        default_factory=lambda: np.empty(0),
        init_var=False,
        repr=False,
    )
    mesh_eigenvalues: npt.NDArray[np.float_] = pydantic.Field(
        default_factory=lambda: np.empty(0),
        init_var=False,
        repr=False,
    )
    mesh_region: npt.NDArray[np.bool_] = pydantic.Field(
        default_factory=lambda: np.empty(0, dtype=np.bool_),
        init_var=False,
        repr=False,
    )
    vertices: npt.NDArray[np.float_] = pydantic.Field(
        default_factory=lambda: np.empty(0),
        init_var=False,
        repr=False,
    )

    def __post_init__(self: typing_extensions.Self) -> None:
        mesh_config = sleplet._mesh_methods.extract_mesh_config(self.name)
        self._camera_view = sleplet._plotly_methods.create_camera(
            mesh_config["CAMERA_X"],
            mesh_config["CAMERA_Y"],
            mesh_config["CAMERA_Z"],
            mesh_config["REGION_ZOOM"] if self.zoom else mesh_config["DEFAULT_ZOOM"],
            x_center=mesh_config["CENTER_X"] if self.zoom else 0,
            y_center=mesh_config["CENTER_Y"] if self.zoom else 0,
            z_center=mesh_config["CENTER_Z"] if self.zoom else 0,
        )
        self._colourbar_pos = (
            mesh_config["REGION_COLOURBAR_POS"]
            if self.zoom
            else mesh_config["DEFAULT_COLOURBAR_POS"]
        )
        self.vertices, self.faces = sleplet._mesh_methods.read_mesh(mesh_config)
        self.mesh_region = sleplet._mesh_methods.create_mesh_region(
            mesh_config,
            self.vertices,
        )
        (
            self.mesh_eigenvalues,
            self.basis_functions,
            self.number_basis_functions,
        ) = sleplet._mesh_methods.mesh_eigendecomposition(
            self.name,
            self.vertices,
            self.faces,
            number_basis_functions=self.number_basis_functions,
        )
