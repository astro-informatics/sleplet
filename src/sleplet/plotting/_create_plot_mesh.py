from dataclasses import KW_ONLY
from pathlib import Path

import cmocean
import numpy as np
import plotly.offline as py
from matplotlib.colors import LinearSegmentedColormap
from numpy import typing as npt
from plotly.graph_objs import Figure, Mesh3d
from plotly.graph_objs.mesh3d import Lighting
from pydantic.dataclasses import dataclass

import sleplet
import sleplet._mask_methods
import sleplet._mesh_methods
import sleplet._plotly_methods
import sleplet._validation
import sleplet.meshes.mesh
import sleplet.plot_methods

_fig_path = Path(__file__).resolve().parents[1] / "_figures"

MESH_CBAR_LEN = 0.95
MESH_CBAR_FONT_SIZE = 32
MESH_UNSEEN = -1e5  # kaleido bug


@dataclass(config=sleplet._validation.Validation)
class PlotMesh:
    """creates surface mesh plot via `plotly`"""

    mesh: sleplet.meshes.mesh.Mesh
    """TODO"""
    filename: str
    """TODO"""
    f: npt.NDArray[np.complex_ | np.float_]
    """TODO"""
    _: KW_ONLY
    amplitude: float | None = None
    """TODO"""
    colour: LinearSegmentedColormap = cmocean.cm.ice
    """TODO"""
    normalise: bool = True
    """TODO"""
    region: bool = False
    """TODO"""

    def __post_init_post_parse__(self) -> None:
        if self.normalise:
            self.filename += "_norm"

    def execute(self) -> None:
        """TODO creates 3d plotly mesh plot"""
        vmin, vmax = self.f.min(), self.f.max()
        f = self._prepare_field(self.f)

        if self.region:
            # make plot area clearer
            f = self._set_outside_region_to_minimum(f)

        # pick largest tick max value
        tick_mark = sleplet._plotly_methods.create_tick_mark(
            vmin, vmax, amplitude=self.amplitude
        )

        data = [
            Mesh3d(
                x=self.mesh.vertices[:, 0],
                y=self.mesh.vertices[:, 2],
                z=self.mesh.vertices[:, 1],
                i=self.mesh.faces[:, 0],
                j=self.mesh.faces[:, 1],
                k=self.mesh.faces[:, 2],
                intensitymode="cell",
                intensity=f,
                cmax=1 if self.normalise else tick_mark,
                cmid=0.5 if self.normalise else 0,
                cmin=0 if self.normalise else -tick_mark,
                colorbar=sleplet._plotly_methods.create_colour_bar(
                    tick_mark,
                    normalise=self.normalise,
                    bar_len=MESH_CBAR_LEN,
                    bar_pos=self.mesh.colourbar_pos,
                    font_size=MESH_CBAR_FONT_SIZE,
                ),
                colorscale=sleplet.plot_methods._convert_colourscale(self.colour),
                lighting=Lighting(ambient=1),
                reversescale=True,
            )
        ]

        layout = sleplet._plotly_methods.create_layout(self.mesh.camera_view)

        fig = Figure(data=data, layout=layout)

        html_filename = str(_fig_path / "html" / f"{self.filename}.html")

        py.plot(fig, filename=html_filename)

        for file_type in {"png", "pdf"}:
            filename = str(_fig_path / file_type / f"{self.filename}.{file_type}")
            sleplet.logger.info(f"saving {filename}")
            fig.write_image(filename, engine="kaleido")

    def _prepare_field(
        self, f: npt.NDArray[np.complex_ | np.float_]
    ) -> npt.NDArray[np.float_]:
        """
        scales the field before plotting
        """
        return sleplet.plot_methods._normalise_function(
            sleplet._mesh_methods.average_functions_on_vertices_to_faces(
                self.mesh.faces, f
            ),
            normalise=self.normalise,
        )

    def _set_outside_region_to_minimum(
        self, f: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """
        for the Slepian region set the outisde area to negative infinity
        hence it is clear we are only interested in the coloured region
        """
        region_on_faces = sleplet._mask_methods.convert_region_on_vertices_to_faces(
            self.mesh
        )
        return np.where(region_on_faces, f, MESH_UNSEEN)
